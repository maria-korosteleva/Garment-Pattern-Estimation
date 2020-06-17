# Training loop func
import numpy as np
import os
import time

import torch
import torch.nn as nn
import wandb as wb

class Trainer():
    def __init__(self, data_name, project_name='Train', run_name='Run', no_sync=False):
        """Initialize training"""
        self.project = project_name
        self.run_name = run_name
        # default training setup
        self.setup = dict(
            random_seed=int(time.time()),
            dataset=data_name,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            epochs=100,
            batch_size=64,
            learning_rate=0.001,
            loss='MSELoss',
            optimizer='SGD',
            lr_scheduling=True
        )

        self.no_sync = no_sync
   
    def init_randomizer(self):
        """Init randomizatoin for torch globally for reproducibility"""
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(self.setup['random_seed'])
        if 'cuda' in self.setup['device']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def fit(self, model, train_loader, valid_loader, run_name=''):

        if run_name:
            self.run_name = run_name
        self.setup['model'] = model.__class__.__name__

        self._add_optimizer(model)
        self._add_loss()
        self._add_scheduler()

        self._init_wb_run(model)

        self.device = torch.device(wb.config.device)
        print('NN training Using device: {}'.format(self.device))
        
        wb.save('checkpoint*.pth')  # upload checkpoints as they are created https://docs.wandb.com/library/save
        self._fit_loop(model, train_loader, valid_loader)

        self._save_final(model)
        print ("Trainer::Finished training")

    def update_config(self, **kwargs):
        """add given values to training config"""
        self.setup.update(kwargs)

    # ---- Private -----
    def _init_wb_run(self, model):
        # init Weights&biases run
        if self.no_sync:
            os.environ['WANDB_MODE'] = 'dryrun'  # No sync with cloud
        wb.init(name=self.run_name, project=self.project, config=self.setup)
        wb.watch(model, log='all')

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
            }, os.path.join(wb.run.dir, 'checkpoint_{}.pth'.format(epoch)))
    
    def _save_final(self, model):
        """Save full model for future independent inference"""

        torch.save(model, os.path.join(wb.run.dir, 'model.pth'))