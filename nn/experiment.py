import os
from pathlib import Path
import requests
import time

import torch
import wandb as wb


class WandbRunWrappper(object):
    """Class provides 
        * a convenient way to store wandb run info 
        * some functions & params shortcuts to access wandb functionality
        * for implemented functions, transparent workflow for finished and active (initialized) run 

        Wrapper currently does NOT wrap one-liners routinely called for active runs like wb.log(), wb.watch()  
    """
    def __init__(self, wandb_username, project_name='Train', run_name='Run', run_id=None, no_sync=False):
        """Init experiment tracking with wandb
            With no_sync==True, run won't sync with wandb cloud. 
            Note that resuming won't work for off-cloud runs as it requiers fetching files from the cloud"""

        self.checkpoint_filetag = 'checkpoint'
        self.final_filetag = 'fin_model_state'
        self.wandb_username = wandb_username
        
        self.project = project_name
        self.run_name = run_name
        self.run_id = run_id
        self.no_sync = False

        # cannot use wb.config, wb.run, etc. until run initialized & logging started & local path
        self.initialized = False  
        self.artifact = None

    # ----- start&stop ------
    def init_run(self, config={}):
        """Start wandb logging. 
            If run_id is known, run is automatically resumed.
            """
        if self.no_sync:
            os.environ['WANDB_MODE'] = 'dryrun'
            print('Experiment:Warning: run is not synced with wandb cloud')

        wb.init(name=self.run_name, project=self.project, config=config, resume=self.run_id)
        self.run_id = wb.run.id

        self.initialized = True
        self.checkpoint_counter = 0

    def stop(self):
        """Stop wandb for current run. All logging finishes & files get uploaded"""
        if self.initialized:
            wb.finish()
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
        return run.config['data_split'], run.config['batch_size'], run.config['dataset']

    def NN_config(self):
        """Run configuration params of NeuralNetwork model"""
        run = self._run_object()
        return run.config['NN']

    def add_statistic(self, tag, info):
        """Add info the run summary (e.g. stats on test set)"""
        # different methods for on-going & finished runs
        if self.initialized:
            wb.run.summary[tag] = info
        else:
            run = self._run_object()
            run.summary[tag] = info
            run.summary.update()

    def add_config(self, tag, info):
        """Add new value to run config. Only for ongoing runs!"""
        if self.initialized:
            wb.config[tag] = info
        else:
            raise RuntimeError('WandbRunWrappper:Error:Cannot add config to finished run')

    def add_artifact(self, path, name, type):
        """Create a new wandb artifact and upload all the contents there"""

        path = Path(path)

        if not self.initialized:
            # can add artifacts only to existing runs!
            # https://github.com/wandb/client/issues/1491#issuecomment-726852201
            print('Experiment::Reactivating wandb run to upload an artifact {}!'.format(name))
            wb.init(id=self.run_id, project=self.project, resume='allow')

        artifact = wb.Artifact(name, type=type)
        if path.is_file():
            artifact.add_file(str(path))
        else:
            artifact.add_dir(str(path))
                    
        wb.run.log_artifact(artifact)

        if not self.initialized:
            wb.finish()

    def is_finished(self):
        run = self._run_object()
        return run.state == 'finished'

    # ---- file info -----
    def checkpoint_filename(self, check_id=None):
        """Produce filename for the checkpoint of given epoch"""
        check_id_str = '_{}'.format(check_id) if check_id is not None else ''
        return '{}{}.pth'.format(self.checkpoint_filetag, check_id_str)

    def artifactname(self, tag, with_version=True, version=None, custom_alias=None):
        """Produce name for wandb artifact for current run with fiven tag"""
        basename = self.run_name + '_' + self.run_id + '_' + tag
        if custom_alias is not None:
            return basename + ':' + custom_alias

        # else -- return a name with versioning
        version_tag = ':v' + str(version) if version is not None else ':latest'
        return basename + version_tag if with_version else basename

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
        # if self.initialized:
        return Path(wb.run.dir)
        # raise RuntimeError('WbRunWrapper:Error:No local path exists: run is not initialized')

    def local_artifact_path(self):
        """create & maintain path to save files to-be-commited-as-artifacts"""
        path = Path('./wandb') / 'artifacts' / self.run_id
        if not path.exists():
            path.mkdir(parents=True)
        return path

    # ----- working with files -------
    def load_checkpoint_file(self, to_path=None, version=None):
        """Load checkpoint file for given epoch from the cloud"""
        if not self.run_id:
            raise RuntimeError('WbRunWrapper:Error:Need to know run id to restore checkpoint from the could')
        try:
            art_path = self._load_artifact(self.artifactname('checkpoint', version=version))
            for file in art_path.iterdir():
                return torch.load(file)
                # only one file per checkpoint anyway

        except (RuntimeError, requests.exceptions.HTTPError, wb.apis.CommError) as e:  # raised when file is corrupted or not found
            print('WbRunWrapper::Error::checkpoint from version \'{}\'is corrupted or lost: {}'.format(version if version else 'latest', e))
            raise e
    
    def load_final_model(self, to_path=None):
        """Load final model parameters file from the cloud if it exists"""
        if not self.run_id:
            raise RuntimeError('WbRunWrapper:Error:Need to know run id to restore final model from the could')
        try:
            art_path = self._load_artifact(self.artifactname('checkpoint'), to_path=to_path)  # loading latest
            for file in art_path.iterdir():
                print(file)
                return torch.load(file)

        except (requests.exceptions.HTTPError, wb.apis.CommError):  # file not found
            raise RuntimeError('WbRunWrapper:Error:No file with final weights found in run {}'.format(self.cloud_path()))
    
    def load_best_model(self, to_path=None):
        """Load model parameters (model with best performance) file from the cloud if it exists"""
        if not self.run_id:
            raise RuntimeError('WbRunWrapper:Error:Need to know run id to restore final model from the could')
        try:
            # best checkpoints have 'best' alias
            art_path = self._load_artifact(self.artifactname('checkpoint', custom_alias='best'), to_path=to_path)
            for file in art_path.iterdir():
                print(file)
                return torch.load(file)
                # only one file per checkpoint anyway

        except (requests.exceptions.HTTPError):  # file not found
            raise RuntimeError('WbRunWrapper:Error:No file with best weights found in run {}'.format(self.cloud_path()))
    
    def save_checkpoint(self, state, aliases=[], wait_for_upload=False):
        """Save given state dict as torch checkpoint to local run dir
            aliases assign labels to checkpoints for easy retrieval
        """

        if not self.initialized:
            # prevent training updated to finished runs
            raise RuntimeError('Experiment::cannot save checkpoint files to non-active wandb runs')

        print('Experiment::Saving model state -- checkpoint artifact')

        # Using artifacts to store important files for this run
        filename = self.checkpoint_filename(self.checkpoint_counter)
        artifact = wb.Artifact(self.artifactname('checkpoint', with_version=False), type='checkpoint')
        self.checkpoint_counter += 1  # ensure all checkpoints have unique names 

        torch.save(state, self.local_artifact_path() / filename)
        artifact.add_file(str(self.local_artifact_path() / filename))
        wb.run.log_artifact(artifact, aliases=['latest'] + aliases)

        if wait_for_upload:
            self._wait_for_upload(self.artifactname('checkpoint', version=self.checkpoint_counter-1))

    # ------- utils -------
    def _load_artifact(self, artifact_name, to_path=None):
        """Download a requested artifact withing current project. Return loaded path"""
        print('Experiment::Requesting artifacts: {}'.format(artifact_name))

        api = wb.Api({'project': self.project})
        artifact = api.artifact(name=artifact_name)
        filepath = artifact.download(str(to_path) if to_path else None)
        print('Experiment::Artifact saved to: {}'.format(filepath))

        return Path(filepath)

    def _run_object(self):
        """ Shortcut for getting reference to wandb api run object. 
            To uniformly access both ongoing & finished runs"""
        return wb.Api().run(self.cloud_path())

    def _wait_for_upload(self, artifact_name, max_attempts=10):
        """Wait for an upload of the given version of an artifact"""
        # follows the suggestion of https://github.com/wandb/client/issues/1486#issuecomment-726229978
        print('Experiment::Waiting for artifact {} upload'.format(artifact_name))
        attempt = 1
        while attempt <= max_attempts:
            try:
                time.sleep(5)
                self._load_artifact(artifact_name)
                print('Requested version is successfully syncronized')
                break
            except (ValueError, wb.errors.error.CommError):
                attempt += 1
                print('Trying again')
        if attempt > max_attempts:
            print('Experiment::Warning::artifact {} is still not syncronized'.format(artifact_name))
