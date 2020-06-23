import os
from pathlib import Path
import requests

import torch
import wandb as wb

class WandbRunWrappper(object):
    def __init__(self, wandb_username, project_name='Train', run_name='Run', run_id=None):
        self.checkpoint_filetag = 'checkpoint'
        self.final_filetag = 'fin_model_state'
        self.wandb_username = wandb_username
        
        self.project = project_name
        self.run_name = run_name
        self.run_id = run_id

        # cannot use wb.config, wb.run, etc. until run initialized & logging started & local path
        self.initialized = False  

    # ----- start&stop ------
    def init_run(self, config={}, no_sync=False):
        """Start wandb logging. 
            If run_id is given, run is automatically resumed.
            
            With no_sync==True, run won't sync with wandb cloud. 
            Note that resuming won't work for off-cloud runs as it requiers fetching files from the cloud
            """
        if no_sync:
            os.environ['WANDB_MODE'] = 'dryrun'
        wb.init(name=self.run_name, project=self.project, config=config, resume=self.run_id)
        self.run_id = wb.run.id
        # upload checkpoints as they are created https://docs.wandb.com/library/save
        wb.save('*' + self.checkpoint_filetag + '*')  

        self.initialized = True

    def stop(self):
        """Stop wandb for current run. All logging finishes & files get uploaded"""
        if self.initialized:
            wb.join()
        self.initialized = False

    # -------- run info ------
    def last_epoch(self):
        """Id of the last epoch processed"""
        run = self._run_object()
        return run.summary['epoch']

    def data_info(self):
        """Info on the data setup from the run config:
            Split & batch size info """
        run = self._run_object()
        return run.config['data_split'], run.config['batch_size']

    # ---- file info -----
    def checkpoint_filename(self, epoch):
        """Produce filename for the checkpoint of given epoch"""
        return '{}_{}.pth'.format(self.checkpoint_filetag, epoch)

    def final_filename(self):
        """Produce filename for the final model file (assuming PyTorch)"""
        return self.final_filetag + '.pth'
    
    def cloud_path(self):
        """Return virtual path to the current run on wandb could
            Implemented as a function to allow dynamic update of components with less bugs =)
        """
        if not self.run_id:
            raise RuntimeError('WbRunWrapper:Error:Need to know run id to get path in wandb could')
        return self.wandb_username + '/' + self.project + '/' + self.run_id

    def local_path(self):
        if self.initialized:
            return Path(wb.run.dir)
        raise RuntimeError('WbRunWrapper:Error:No local path exists: run is not initialized')

    # ----- working with files -------
    def load_checkpoint_file(self, epoch):
        """Load checkpoint file for given epoch from the cloud"""
        if not self.run_id:
            raise RuntimeError('WbRunWrapper:Error:Need to know run id to restore files from the could')
        try:
            print('Requesting {}'.format(self.checkpoint_filename(epoch)))
            wb.restore(self.checkpoint_filename(epoch), run_path=self.cloud_path())
            # https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208
            checkpoint = torch.load(self.local_path() / self.checkpoint_filename(epoch))
            return checkpoint
        except (RuntimeError, requests.exceptions.HTTPError, wb.apis.CommError) as e:  # raised when file is corrupted or not found
            print('WbRunWrapper:Warning:checkpoint from epoch {} is corrupted or lost: {}'.format(epoch, e))
            return None
    
    def load_final_model(self):
        """Load final model parameters file from the cloud if it exists"""
        if not self.run_id:
            raise RuntimeError('WbRunWrapper:Error:Need to know run id to restore final model from the could')
        try:
            wb.restore(self.final_filename, run_path=self.cloud_path())
        except (requests.exceptions.HTTPError, wb.apis.CommError):  # file not found
            print('WbRunWrapper:Warning:No file with final weights found in run {}'.format(self.cloud_path()))
            return None
    
    def save(self, state, epoch=None, final=False, filename=''):
        """Save given state dict as torch checkpoint to local run dir
            epoch/final/filename parameters control saving names (multiple could be used):
            * If epoch parameter is given, state is saved as checkpoint file for given epoch
            * If final is true, state is saved as final model file
            * if filename is given, state is simply saved with ginev filename"""
        if epoch is not None:
            torch.save(state, self.local_path() / self.checkpoint_filename(epoch))
        if final:
            torch.save(state, self.local_path() / self.final_filename())
        if filename:
            torch.save(state, self.local_path() / filename)

    # ------- utils -------
    def _run_object(self):
        """ Shortcut for getting reference to wandb api run object. 
            This object allows to access both ongoing & finished runs"""
        return wb.Api().run(self.cloud_path())