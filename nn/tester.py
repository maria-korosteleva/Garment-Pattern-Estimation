import torch
import torch.nn as nn


class Tester():
    """Forward passes on already trained models"""
    def __init__(self, model, wandb_run_id=None):
        self.model = model
        self.regression_loss = nn.MSELoss()
        self.data_section_list = ['full', 'validation', 'test']
        pass

    @staticmethod
    def from_wandb(run_id, from_checkpoint=False):
        """Retrieve model & data split info from wandb run"""
        pass

    def metrics(self, data_wrapper, section='test'):
        """Evalutes avalible metrics on the given dataset section"""
        if section not in self.data_section_list:
            raise ValueError('Tester::requested evaluation on unknown data section {}'.format(section))

        # TODO provide dataset not wrapper + split
        
        self.model.eval()
        with torch.no_grad():
            if section == 'validation':
                loss = sum([self.regression_loss(self.model(batch['features']), batch['pattern_params']) for batch in data_wrapper.loader_validation]) 
            else: 
                loss = None
        
        return {'loss': loss}

    def predict(self, data_wrapper, section='test', single_batch=False):
        """Save model predictions on the given dataset section"""
        if section not in self.data_section_list:
            raise ValueError('Tester::requested evaluation on unknown data section {}'.format(section))

        self.model.eval()
        with torch.no_grad():
            if section == 'validation' and single_batch:
                batch = next(iter(data_wrapper.loader_validation))    # might have some issues, see https://github.com/pytorch/pytorch/issues/1917
                data_wrapper.dataset.save_prediction_batch(self.model(batch['features']), batch['name'])

    def one_shot_prediction(self, filename):
        """Predict on the given single datapoint"""
        pass


