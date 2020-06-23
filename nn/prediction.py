import torch
import torch.nn as nn


class PredictionManager():
    """Forward passes on already trained models"""
    def __init__(self):
        self.metric_functions = {
            'regression loss': nn.MSELoss()
        }
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def metrics(self, model, data_wrapper, section='test'):
        """Evalutes avalible metrics on the given dataset section"""
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            current_metrics = {}
            loader = data_wrapper.get_loader(section)
            if loader:
                current_metrics = dict.fromkeys(self.metric_functions, 0)
                for batch in loader:
                    features, params = batch['features'].to(self.device), batch['pattern_params'].to(self.device)
                    for metric in current_metrics:
                        current_metrics[metric] += self.metric_functions[metric](model(features), params)
                # normalize & convert
                for metric in current_metrics:
                    current_metrics[metric] = current_metrics[metric].cpu().numpy()  # conversion only works on cpu
                    current_metrics[metric] /= len(loader)
        
        return current_metrics

    def predict(self, model, data_wrapper, section='test', single_batch=False):
        """Save model predictions on the given dataset section"""
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            loader = data_wrapper.get_loader(section)
            if loader:
                if single_batch:
                    batch = next(iter(loader))    # might have some issues, see https://github.com/pytorch/pytorch/issues/1917
                    features = batch['features'].to(self.device)
                    data_wrapper.dataset.save_prediction_batch(model(features), batch['name'])
                else:
                    for batch in loader:
                        features = batch['features'].to(self.device)
                        data_wrapper.dataset.save_prediction_batch(model(features), batch['name'])

    def one_shot_prediction(self, filename):
        """Predict on the given single datapoint"""
        pass


