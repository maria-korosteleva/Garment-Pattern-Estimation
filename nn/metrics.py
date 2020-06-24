"""List of metrics to evalute on a model and a dataset"""

import torch
import torch.nn as nn


metric_functions = {
    'regression loss': nn.MSELoss()
}

def eval_metrics(model, data_wrapper, section='test'):
    """Evalutes all avalible metrics from metric_functions on the given dataset section"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        current_metrics = {}
        loader = data_wrapper.get_loader(section)
        if loader:
            current_metrics = dict.fromkeys(metric_functions, 0)
            for batch in loader:
                features, params = batch['features'].to(device), batch['ground_truth'].to(device)
                for metric in current_metrics:
                    current_metrics[metric] += metric_functions[metric](model(features), params)
            # normalize & convert
            for metric in current_metrics:
                current_metrics[metric] = current_metrics[metric].cpu().numpy()  # conversion only works on cpu
                current_metrics[metric] /= len(loader)
    
    return current_metrics

