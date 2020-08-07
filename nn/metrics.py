"""List of metrics to evalute on a model and a dataset"""

import torch
import torch.nn as nn

# ------- custom metrics --------
class PanelLoopLoss():
    """Evaluate loss for the panel edge sequence representation property: 
        ensuring edges within panel loop & return to origin"""
    def __init__(self, data_stats={}):
        """Info for evaluating padding vector if data statistical info is applied to it.
            * if standardization/normalization transform is applied to padding, 'data_stats' should be provided
                'data_stats' format: {'shift': <torch.tenzor>, 'scale': <torch.tensor>} 
        """
        self._eval_pad_vector(data_stats)
            
    def __call__(self, predicted_panels, original_panels=None, data_stats={}):
        """Evaluate loop loss on provided predicted_panels batch.
            * 'original_panels' are used to evaluate the correct number of edges of each panel in case padding is applied.
                If 'original_panels' is not given, it is assumed that there is no padding
                If data stats are not provided at init or in this call, zero vector padding is assumed
            * data_stats can be used to update padding vector on the fly
        """
        # flatten input into list of panels
        if len(predicted_panels.shape) > 3:
            predicted_panels = predicted_panels.view(-1, predicted_panels.shape[-2], predicted_panels.shape[-1])
        if original_panels is not None and len(original_panels.shape) > 3:
            original_panels = original_panels.view(-1, original_panels.shape[-2], original_panels.shape[-1])

        # prepare for padding comparison
        if data_stats:
            self._eval_pad_vector(data_stats)
        if original_panels is not None:
            if self.pad_tenzor is None:  # assume zero vector for padding
                self.pad_tenzor = torch.zeros(original_panels.shape[-1])
            pad_tenzor_propagated = self.pad_tenzor.repeat(original_panels.shape[1], 1)
            pad_tenzor_propagated = pad_tenzor_propagated.to(device=predicted_panels.device)
            
        panel_coords_sum = torch.zeros((predicted_panels.shape[0], 2))
        panel_coords_sum = panel_coords_sum.to(device=predicted_panels.device)
        # iterate over elements in batch
        for el_id in range(predicted_panels.shape[0]):
            if original_panels is not None and self.pad_tenzor is not None:
                # panel original length
                panel = original_panels[el_id]
                # unpaded length
                bool_matrix = torch.isclose(panel, pad_tenzor_propagated, atol=1.e-2)
                seq_len = (~torch.all(bool_matrix, axis=1)).sum()  # only non-padded rows
            else:
                seq_len = len(predicted_panels[el_id])

            # get per-coordinate sum of edges endpoints of each panel
            panel_coords_sum[el_id] = predicted_panels[el_id][:seq_len, :2].sum(axis=0)

        panel_square_sums = panel_coords_sum ** 2  # per sum square

        # batch mean of squared norms of per-panel final points:
        return panel_square_sums.sum() / len(panel_square_sums)

    def _eval_pad_vector(self, data_stats={}):
        # prepare padding vector for unpadding the panel data on call
        if data_stats:
            mean = torch.Tensor(data_stats['mean'])
            std = torch.Tensor(data_stats['std'])
            self.pad_tenzor = - mean / std
        else:
            self.pad_tenzor = None

# ------- Metrics evaluation -------------

metric_functions = {
}  # No extra metrics are defined right now

def eval_metrics(model, data_wrapper, section='test', loop_loss=False):
    """Evalutes all avalible metrics from metric_functions on the given dataset section"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    if loop_loss:
        # modify loss object according to the data stats
        if 'standardize' in data_wrapper.dataset.config:
            # NOTE assuming shift&scale is applied to padding
            metric_functions['loop_loss'] = PanelLoopLoss(data_stats=data_wrapper.dataset.config['standardize'])
        else:
            metric_functions['loop_loss'] = PanelLoopLoss()  # no padding == zero padding assumed

    with torch.no_grad():
        current_metrics = {}
        loader = data_wrapper.get_loader(section)
        if loader:
            current_metrics = dict.fromkeys(metric_functions, 0)
            model_defined = 0
            loop_loss = 0
            for batch in loader:
                features, gt = batch['features'].to(device), batch['ground_truth'].to(device)
                # basic metric
                model_defined += model.loss(features, gt)
                # other metrics from this module
                preds = model(features)
                for metric in current_metrics:
                    current_metrics[metric] += metric_functions[metric](preds, gt)
            current_metrics['model_defined'] = model_defined
            # normalize & convert
            for metric in current_metrics:
                current_metrics[metric] = current_metrics[metric].cpu().numpy()  # conversion only works on cpu
                current_metrics[metric] /= len(loader)
    
    return current_metrics
