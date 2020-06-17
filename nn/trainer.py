# Training loop func
import numpy as np
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import wandb as wb

# My modules
import data

class Trainer():
    def __init__(self, project_name='Train', run_name='Run', no_sync=False, allow_resume=False):
        """Initialize training"""
        self.checkpoint_filename = 'last_checkpoint.pth'
        
        self.project = project_name
        self.run_name = run_name
        self.no_sync = no_sync
        self.resume = allow_resume
        self.datawraper = None

        # default training setup
        self.setup = dict(
            model_random_seed=None,
            dataset=None,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            epochs=100,
            batch_size=64,
            learning_rate=0.001,
            loss='MSELoss',
            optimizer='SGD',
            lr_scheduling=True
        )
   
    def init_randomizer(self, random_seed=None):
        """Init randomizatoin for torch globally for reproducibility"""
        # see https://pytorch.org/docs/stable/notes/randomness.html
        if random_seed:
            self.setup['model_random_seed'] = random_seed
        elif not self.setup['model_random_seed']:
            self.setup['model_random_seed'] = int(time.time())

        torch.manual_seed(self.setup['model_random_seed'])
        if 'cuda' in self.setup['device']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def update_config(self, **kwargs):
        """add given values to training config"""
        self.setup.update(kwargs)

    def use_dataset(self, dataset, valid_percent=None, test_percent=None):
        """Use specified dataset for training with given split settings"""
        self.setup['dataset'] = dataset.name
        self.datawraper = data.DatasetWrapper(dataset)
        self.datawraper.new_split(valid_percent, test_percent)
        self.datawraper.new_loaders(self.setup['batch_size'], shuffle_train=True)

        self.update_config(data_split=self.datawraper.split_info)

        print ('{} split: {} / {}'.format(dataset.name, len(self.datawraper.training), len(self.datawraper.validation)))

        return self.datawraper

    def fit(self, model, run_name=''):
        """Fit proveided model to reviosly configured dataset"""
        if not self.datawraper:
            raise RuntimeError('Trainer::Error::fit before dataset was provided. run use_dataset() first')
        if run_name:
            self.run_name = run_name
        self.setup['model'] = model.__class__.__name__

        self._add_optimizer(model)
        self._add_loss()
        self._add_scheduler()

        start_epoch = self._init_wb_run(model)

        self.device = torch.device(wb.config.device)
        print('NN training Using device: {}'.format(self.device))
        
        wb.save('*checkpoint*')  # upload checkpoints as they are created https://docs.wandb.com/library/save
        self._fit_loop(model, self.datawraper.loader_train, self.datawraper.loader_validation, start_epoch=start_epoch)

        self._save_final(model)
        print ("Trainer::Finished training")

    # ---- Private -----
    def _init_wb_run(self, model):
        # init Weights&biases run
        if self.no_sync:
            os.environ['WANDB_MODE'] = 'dryrun'  # No sync with cloud
        print(self.resume)
        wb.init(name=self.run_name, project=self.project, config=self.setup, resume=self.resume)

        if wb.run.resumed:
            # restore last checkpoint
            wb.restore(self.checkpoint_filename)
            checkpoint = torch.load(Path(wb.run.dir) / self.checkpoint_filename)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

            # restore data split -- assuming wb config values are also restored
            self.datawraper.load_split(wb.config.data_split)  # NOTE : random number generator reset
            self.datawraper.new_loaders(wb.config.batch_size)  # should reproduce shuffle before resume
            
            # TODO restore config??? 
        else:
            start_epoch = 0

        wb.watch(model, log='all')
        return start_epoch

    def _add_optimizer(self, model):
        
        if self.setup['optimizer'] == 'SGD':
            # future 'else'
            print('NN Warning::Using default SGD optimizer')
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.setup['learning_rate'])
        
    def _add_loss(self):
        if self.setup['loss'] == 'MSELoss':
            # future 'else'
            print('NN Warning::Using default MSELoss loss')
            self.regression_loss = nn.MSELoss()

    def _add_scheduler(self):
        if ('lr_scheduling' in self.setup
                and self.setup['lr_scheduling']):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1)
        else:
            print('NN Warning: no learning scheduling set')

    def _fit_loop(self, model, train_loader, valid_loader, start_epoch=0):
        """Fit loop with the setup already performed"""
        model.to(self.device)
        for epoch in range (start_epoch, wb.config.epochs):
            model.train()
            for i, batch in enumerate(train_loader):
                features, params = batch['features'].to(self.device), batch['pattern_params'].to(self.device)
                
                #with torch.autograd.detect_anomaly():
                preds = model(features)
                loss = self.regression_loss(preds, params)
                #print ('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss))
                loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # logging
                if i % 5 == 4:
                    wb.log({'epoch': epoch, 'loss': loss})
            # checkpoint
            self._save_checkpoint(model, epoch)

            # scheduler step
            model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[(self.regression_loss(model(features), params), len(batch)) for batch in valid_loader]
                )                
            valid_loss = np.sum(losses) / np.sum(nums)
            self.scheduler.step(valid_loss)
            
            # little logging
            print ('Epoch: {}, Validation Loss: {}'.format(epoch, valid_loss))
            wb.log({'epoch': epoch, 'valid_loss': valid_loss, 'learning_rate': self.optimizer.param_groups[0]['lr']})

    def _save_checkpoint(self, model, epoch):
        """Save checkpoint to be used to resume training"""
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, Path(wb.run.dir) / self.checkpoint_filename)
    
    def _save_final(self, model):
        """Save full model for future independent inference"""

        torch.save(model, Path(wb.run.dir) / 'model.pth')