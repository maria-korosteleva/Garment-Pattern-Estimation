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
import data as data

class Trainer():
    def __init__(self, experiment_tracker, dataset=None, valid_percent=None, test_percent=None, split_seed=None, with_norm=True, with_visualization=False):
        """Initialize training and dataset split (if given)
            * with_visualization toggles image prediction logging to wandb board. Only works on custom garment datasets (with prediction -> image) conversion"""
        self.experiment = experiment_tracker
        self.datawraper = None
        self.use_data_normalization = True
        self.log_with_visualization = with_visualization
        
        # default training setup
        self.setup = dict(
            model_random_seed=None,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            epochs=2000,
            batch_size=32,
            learning_rate=0.001,
            optimizer='Adam',
            weight_decay=0,
            lr_scheduling={
                'patience': 10,
                'factor': 0.5
            },
            early_stopping={
                'window': 0.005,
                'patience': 50
            }
        )

        if dataset is not None:
            self.use_dataset(dataset, valid_percent, test_percent, split_seed)
   
    def init_randomizer(self, random_seed=None):
        """Init randomizatoin for torch globally for reproducibility. 
            Using this function ensures that random seed will be recorded in config
        """
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

    def use_dataset(self, dataset, valid_percent=None, test_percent=None, random_seed=None):
        """Use specified dataset for training with given split settings"""
        self.datawraper = data.DatasetWrapper(dataset)
        self.datawraper.new_split(valid_percent, test_percent, random_seed)
        self.datawraper.new_loaders(self.setup['batch_size'], shuffle_train=True)

        # ---- NEW! -----
        if self.use_data_normalization:
            self.datawraper.use_normalization()

        return self.datawraper

    def fit(self, model):
        """Fit provided model to reviosly configured dataset"""
        if not self.datawraper:
            raise RuntimeError('Trainer::Error::fit before dataset was provided. run use_dataset() first')
        self.setup['model'] = model.__class__.__name__

        self.device = self.setup['device']
    
        self._add_optimizer(model)
        self._add_scheduler()
        self.es_tracking = []  # early stopping init

        start_epoch = self._start_experiment(model)
        print('Trainer::NN training Using device: {}'.format(self.device))

        if self.log_with_visualization:
            # go to upper dir to avoid bug TODO https://github.com/wandb/client/issues/1128
            self.folder_for_preds = Path(wb.run.dir) / '..' / 'intermediate_preds'
            self.folder_for_preds.mkdir(exist_ok=True)
        
        self._fit_loop(model, self.datawraper.loader_train, self.datawraper.loader_validation, start_epoch=start_epoch)

        self.experiment.save(model.state_dict(), save_name='final')
        print ("Trainer::Finished training")

    # ---- Private -----
    def _fit_loop(self, model, train_loader, valid_loader, start_epoch=0):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""
        model.to(self.device)
        log_step = wb.run.step - 1
        
        for epoch in range (start_epoch, wb.config.epochs):
            model.train()
            for i, batch in enumerate(train_loader):
                features, gt = batch['features'].to(self.device), batch['ground_truth'].to(self.device)
                
                #with torch.autograd.detect_anomaly():
                loss = model.loss(features, gt)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # logging
                log_step += 1
                wb.log({'epoch': epoch, 'batch': i, 'loss': loss}, step=log_step)

            # scheduler step: after optimizer step, see https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            model.eval()
            with torch.no_grad():
                losses = [model.loss(batch['features'].to(self.device), batch['ground_truth'].to(self.device)) for batch in valid_loader]
            valid_loss = np.sum(losses) / len(losses)  # Each loss element is already a meacn for its batch
            self.scheduler.step(valid_loss)
            
            # Base logging
            print ('Epoch: {}, Validation Loss: {}'.format(epoch, valid_loss))
            wb.log({'epoch': epoch, 'valid_loss': valid_loss, 'learning_rate': self.optimizer.param_groups[0]['lr']}, step=log_step)

            # prediction for visual reference
            if self.log_with_visualization:
                self._log_an_image(model, valid_loader, epoch, log_step)

            # checkpoint
            self._save_checkpoint(model, epoch)

            # check for early stoping
            if self._early_stopping(loss, valid_loss):
                print('Trainer::Stopped training early')
                break

    def _start_experiment(self, model):
        self.experiment.init_run(self.setup)

        if wb.run.resumed:
            start_epoch = self._restore_run(model)
            print('Trainer::Resumed run {} from epoch {}'.format(self.experiment.cloud_path(), start_epoch))

            if self.device != wb.config.device:
                # device doesn't matter much, so we just inform but do not crash
                print('Trainer::Warning::Resuming run on different device. Was {}, now using {}'.format(wb.config.device, self.device))
        else:
            start_epoch = 0
            self.datawraper.save_to_wandb(self.experiment)

        wb.watch(model, log='all')
        return start_epoch

    def _add_optimizer(self, model):
        
        if self.setup['optimizer'] == 'SGD':
            # future 'else'
            print('Trainer::Using default SGD optimizer')
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.setup['learning_rate'], weight_decay=self.setup['weight_decay'])
        elif self.setup['optimizer'] == 'Adam':
            # future 'else'
            print('Trainer::Using Adam optimizer')
            model.to(self.setup['device'])  # see https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165/8
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.setup['learning_rate'], weight_decay=self.setup['weight_decay'])

    def _add_scheduler(self):
        if 'lr_scheduling' in self.setup:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.setup['lr_scheduling']['factor'], patience=self.setup['lr_scheduling']['patience'], verbose=True)
        else:
            print('Trainer::Warning::no learning scheduling set')

    def _restore_run(self, model):
        """Restore the training process from the point it stopped at. 
            Assuming 
                * Current wb.config state is the same as it was when run was initially created
                * All the necessary training objects are already created and only need update
                * All related object types are the same as in the resuming run (model, optimizer, etc.)
                * Self.run_id is properly set
            Returns id of the next epoch to resume from. """
        
        # data split
        split, batch_size, data_config = self.experiment.data_info()

        self.datawraper.dataset.update_config(data_config)
        self.datawraper.load_split(split, batch_size)  # NOTE : random number generator reset

        # get latest checkoint info
        print('Trainer::Loading checkpoint to resume run..')
        checkpoint = self.experiment.load_checkpoint_file()  # latest

        # checkpoint loaded correctly
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208

        # new epoch id
        return checkpoint['epoch'] + 1

    def _early_stopping(self, last_loss, last_tracking_loss):
        """Check if conditions are met to stop training. Returns a message with a reason if met
            Early stopping allows to save compute time"""

        # loss goes into nans
        if torch.isnan(last_loss):
            print('Trainer::EarlyStopping::Detected nan training losses')
            self.experiment.add_statistic('stopped early', 'Nan in losses')
            return True

        # Target metric is not improving for some time
        self.es_tracking.append(last_tracking_loss.item())
        if len(self.es_tracking) > (wb.config.early_stopping['patience'] + 1):  # number of last calls to consider plus current -> at least two
            self.es_tracking.pop(0)
            # if all values fit into a window, they don't change much
            if abs(max(self.es_tracking) - min(self.es_tracking)) < wb.config.early_stopping['window']:
                print('Trainer::EarlyStopping::Metric have not changed for {} epochs'.format(wb.config.early_stopping['patience']))
                self.experiment.add_statistic('stopped early', 'Metric have not changed for {} epochs'.format(wb.config.early_stopping['patience']))
                return True
        # do not check untill wb.config.early_stopping.patience # of calls are gathered
        
        return False

    def _log_an_image(self, model, loader, epoch, log_step):
        """Log image of one example prediction to wandb.
            If the loader does not shuffle batches, logged image is the same on every step"""
        with torch.no_grad():
            for batch in loader:
                img_files = self.datawraper.dataset.save_prediction_batch(
                    model(batch['features'].to(self.device)), batch['name'], save_to=self.folder_for_preds)
                
                print('Trainer::Logged pattern prediction for {}'.format(img_files[0].name))
                wb.log({batch['name'][0]: wb.Image(str(img_files[0])), 'epoch': epoch}, step=log_step)  # will raise errors if given file is not an image
                break  # One is enough

    def _save_checkpoint(self, model, epoch):
        """Save checkpoint to be used to resume training"""
        self.experiment.save(
            {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
            },
            save_name='checkpoint'
        )
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training