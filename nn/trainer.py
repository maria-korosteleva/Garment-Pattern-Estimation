# Training loop func
import numpy as np
import os
from pathlib import Path
import requests
import time

import torch
import torch.nn as nn
import wandb as wb

# My modules
import data
from experiment import WandbRunWrappper

class Trainer():
    def __init__(self, wandb_username, project_name='Train', run_name='Run', no_sync=False, resume_run_id=None):
        """Initialize training"""
        self.experiment = WandbRunWrappper(
            wandb_username, project_name, run_name, resume_run_id
        )
        self.no_sync = no_sync
        self.datawraper = None

        # default training setup
        self.setup = dict(
            model_random_seed=None,
            dataset=None,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            epochs=5,
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

    def fit(self, model):
        """Fit proveided model to reviosly configured dataset"""
        if not self.datawraper:
            raise RuntimeError('Trainer::Error::fit before dataset was provided. run use_dataset() first')

        self.setup['model'] = model.__class__.__name__

        self._add_optimizer(model)
        self._add_loss()
        self._add_scheduler()

        start_epoch = self._start_experiment(model)

        self.device = torch.device(wb.config.device)
        print('NN training Using device: {}'.format(self.device))
        
        self._fit_loop(model, self.datawraper.loader_train, self.datawraper.loader_validation, start_epoch=start_epoch)

        self.experiment.save(model.state_dict(), final=True)
        print ("Trainer::Finished training")

    # ---- Private -----
    def _start_experiment(self, model):
        self.experiment.init_run(self.setup, self.no_sync)

        if wb.run.resumed:
            start_epoch = self._restore_run(model)

            print('Trainer: Resumed run {} from epoch {}'.format(self.experiment.cloud_path(), start_epoch))
        else:
            start_epoch = 0

        wb.watch(model, log='all')
        return start_epoch

    def _add_optimizer(self, model):
        
        if self.setup['optimizer'] == 'SGD':
            # future 'else'
            print('Trainer::Warning::Using default SGD optimizer')
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.setup['learning_rate'])
        
    def _add_loss(self):
        if self.setup['loss'] == 'MSELoss':
            # future 'else'
            print('Trainer::Warning::Using default MSELoss loss')
            self.regression_loss = nn.MSELoss()

    def _add_scheduler(self):
        if ('lr_scheduling' in self.setup
                and self.setup['lr_scheduling']):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1)
        else:
            print('Trainer::Warning::no learning scheduling set')

    def _fit_loop(self, model, train_loader, valid_loader, start_epoch=0):
        """Fit loop with the setup already performed"""
        model.to(self.device)
        log_step = wb.run.step - 1
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
                    log_step += 1
                    wb.log({'epoch': epoch, 'batch': i, 'loss': loss}, step=log_step)

            # scheduler step: after optimizer step, see https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            model.eval()
            with torch.no_grad():
                losses = [self.regression_loss(model(batch['features'].to(self.device)), batch['pattern_params'].to(self.device)) for batch in valid_loader]
            valid_loss = np.sum(losses) / len(losses)  # Each loss element is already a meacn for its batch
            self.scheduler.step(valid_loss)
            
            # little logging
            print ('Epoch: {}, Validation Loss: {}'.format(epoch, valid_loss))
            wb.log({'epoch': epoch, 'valid_loss': valid_loss, 'learning_rate': self.optimizer.param_groups[0]['lr']}, step=log_step)

            # checkpoint
            self._save_checkpoint(model, epoch)

    def _restore_run(self, model):
        """Restore the training process from the point it stopped at. 
            Assuming 
                * current wb.config state is the same as it was when run was initially created
                * all the necessary training objects are already created and only need update
                * self.resume_run_id is properly set
            Returns id of the next epoch to resume from. """
        
        # data split
        split, batch_size = self.experiment.data_info()
        self.datawraper.load_split(split)  # NOTE : random number generator reset
        self.datawraper.new_loaders(batch_size)  # should reproduce shuffle before resume

        # get latest checkoint info
        print('Trying to load checkpoint..')
        last_epoch = self.experiment.last_epoch()
        # look for last uncorruted checkpoint
        while last_epoch >= 0:
            checkpoint = self.experiment.load_checkpoint_file(last_epoch)
            if checkpoint:
                break  # successfull load
            last_epoch -= 1
        else:
            raise RuntimeError(
                'Trainer::No uncorupted checkpoints found for resuming the run from epoch{}. It\'s recommended to start anew'.format(wb.run.summary['epoch']))
        
        # checkpoint loaded correctly
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208

        # new epoch id
        return checkpoint['epoch'] + 1

    def _save_checkpoint(self, model, epoch):
        """Save checkpoint to be used to resume training"""
        self.experiment.save(
            {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
            },
            epoch=epoch
        )
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training