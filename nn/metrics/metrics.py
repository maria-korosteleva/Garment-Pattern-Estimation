"""
    List of metrics to evalute on a model and a dataset, along with pre-processing methods needed for such evaluation
"""

import torch
import torch.nn as nn

from entmax import SparsemaxLoss  # https://github.com/deep-spin/entmax

import gap

# My modules
from data import Garment3DPatternFullDataset as PatternDataset
from pattern_converter import InvalidPatternDefError


# ------- Model evaluation shortcut -------------
def eval_metrics(model, data_wrapper, section='test'):
    """Evalutes current model on the given dataset section"""
    device = model.device_ids[0] if hasattr(model, 'device_ids') else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()

    loss = model.module.loss if hasattr(model, 'module') else model.loss  # distinguish case of multi-gpu training

    if hasattr(loss, 'with_quality_eval'):
        loss.with_quality_eval = True  # force quality evaluation for losses that support it

    with torch.no_grad():
        loader = data_wrapper.get_loader(section)
        if isinstance(loader, dict):
            metrics_dict = {}
            for data_folder, loader in loader.items():
                metrics_dict[data_folder] = _eval_metrics_per_loader(model, loss, loader, device)
            return metrics_dict
        else:
            return _eval_metrics_per_loader(model, loss, loader, device)


def _eval_metrics_per_loader(model, loss, loader, device):
    """
    Evaluate model on given loader. 
    
    Secondary function -- it assumes that context is set up: torch.no_grad(), model device & mode, etc."""

    current_metrics = dict.fromkeys(['full_loss'], 0)
    counter = 0
    loader_iter = iter(loader)
    while True:  # DEBUG
        try:
            batch = next(loader_iter)  #  loader_iter.next()
        except StopIteration:
            break
        except InvalidPatternDefError as e:
            print(e)
            continue

        features, gt = batch['features'].to(device), batch['ground_truth']
        if gt is None or (hasattr(gt, 'nelement') and gt.nelement() == 0):  # assume reconstruction task
            gt = features

        # loss evaluation
        full_loss, loss_dict, _ = loss(model(features), gt, names=batch['name'])  # use names for cleaner errors when needed

        # summing up
        current_metrics['full_loss'] += full_loss
        for key, value in loss_dict.items():
            if key not in current_metrics:
                current_metrics[key] = 0  # init new metric
            current_metrics[key] += value

    # normalize & convert
    for metric in current_metrics:
        if isinstance(current_metrics[metric], torch.Tensor):
            current_metrics[metric] = current_metrics[metric].cpu().numpy()  # conversion only works on cpu
        current_metrics[metric] /= len(loader)

    return current_metrics


# ----- Utils -----
def eval_pad_vector(data_stats={}):
    # prepare padding vector used for panel padding 
    if data_stats:
        shift = torch.Tensor(data_stats['shift'])
        scale = torch.Tensor(data_stats['scale'])
        return (- shift / scale)
    else:
        return None


# ------- custom quality metrics --------
class PatternStitchPrecisionRecall():
    """Evaluate Precision and Recall scores for pattern stitches prediction
        NOTE: It's NOT a diffentiable evaluation
    """

    def __init__(self, data_stats=None):
        self.data_stats = data_stats
        if data_stats is not None: 
            for key in self.data_stats:
                self.data_stats[key] = torch.Tensor(self.data_stats[key])

    def __call__(
            self, stitch_tags, free_edge_class, gt_stitches, gt_stitches_nums, pattern_names=None):
        """
         Evaluate on the batch of stitch tags
        """
        # undo stats application if provided
        if self.data_stats is not None:
            device = stitch_tags.device
            stitch_tags = stitch_tags * self.data_stats['scale'].to(device) + self.data_stats['shift'].to(device)
        
        tot_precision = 0.
        tot_recall = 0.
        for pattern_idx in range(stitch_tags.shape[0]):
            stitch_list = PatternDataset.tags_to_stitches(stitch_tags[pattern_idx], free_edge_class[pattern_idx]).to(gt_stitches.device)

            num_detected_stitches = stitch_list.shape[1] if stitch_list.numel() > 0 else 0
            if not num_detected_stitches:  # no stitches detected -- zero recall & precision
                continue
            num_actual_stitches = gt_stitches_nums[pattern_idx]
            
            # compare stitches
            correct_stitches = 0.
            for detected in stitch_list.transpose(0, 1):
                for actual in gt_stitches[pattern_idx][:, :gt_stitches_nums[pattern_idx]].transpose(0, 1):
                    # order-invariant comparison of stitch sides
                    correct = (all(detected == actual) or all(detected == actual.flip([0])))
                    correct_stitches += correct
                    if correct:  # no need to check subsequent stitches
                        break
 
                if pattern_names is not None and not correct:  # never detected a match with actual stitches
                    print('StitchPrecisionRecall::{}::Stitch {} detected wrongly'.format(pattern_names[pattern_idx], detected))

            # precision -- how many of the detected stitches are actually there
            tot_precision += correct_stitches / num_detected_stitches if num_detected_stitches else 0.
            # recall -- how many of the actual stitches were detected
            tot_recall += correct_stitches / num_actual_stitches if num_actual_stitches else 0.
        
        # evrage by batch
        return tot_precision / stitch_tags.shape[0], tot_recall / stitch_tags.shape[0]

    def on_loader(self, data_loader, model):
        """Evaluate recall&precision of stitch detection on the full data loader"""

        with torch.no_grad():
            tot_precision = tot_recall = 0
            for batch in data_loader:
                predictions = model(batch['features'])
                batch_precision, batch_recall = self(predictions['stitch_tags'], batch['ground_truth']['stitches'], batch['name'])
                tot_precision += batch_precision
                tot_recall += batch_recall

        return tot_precision / len(data_loader), tot_recall / len(data_loader)


class NumbersInPanelsAccuracies():
    """
        Evaluate in how many cases the number of panels in patterns and number of edges in panels were detected correctly
    """
    def __init__(self, max_edges_in_panel, data_stats=None):
        """
            Requesting data stats to recognize padding correctly
            Should be a dictionary with {'shift': <>, 'scale': <>} keys containing stats for panel outlines
        """
        self.data_stats = data_stats
        self.max_panel_len = max_edges_in_panel
        self.pad_vector = eval_pad_vector(data_stats)
        self.empty_panel_template = self.pad_vector.repeat(self.max_panel_len, 1)

    def __call__(self, predicted_outlines, gt_num_edges, gt_panel_nums, pattern_names=None):
        """
         Evaluate on the batch of panel outlines predictoins 
        """
        batch_size = predicted_outlines.shape[0]
        max_num_panels = predicted_outlines.shape[1]
        if self.empty_panel_template.device != predicted_outlines.device:
            self.empty_panel_template = self.empty_panel_template.to(predicted_outlines.device)

        correct_num_panels = 0.
        num_edges_accuracies = 0.

        correct_pattern_mask = torch.zeros(batch_size, dtype=torch.bool)
        num_edges_in_correct = 0.

        for pattern_idx in range(batch_size):
            # assuming all empty panels are at the end of the pattern, if any
            predicted_num_panels = 0
            correct_num_edges = 0.
            for panel_id in range(max_num_panels):
                predicted_bool_matrix = torch.isclose(
                    predicted_outlines[pattern_idx][panel_id], 
                    self.empty_panel_template, atol=0.07)  # this value is adjusted to have similar effect to what is used in core.py

                # check is the num of edges matches
                predicted_num_edges = (~torch.all(predicted_bool_matrix, axis=1)).sum()  # only non-padded rows
            
                if predicted_num_edges < 3:
                    # 0, 1, 2 edges are not enough to form a panel
                    #  -> assuming this is an empty panel & moving on
                    continue
                # othervise, we have a real panel
                predicted_num_panels += 1

                panel_correct = (predicted_num_edges == gt_num_edges[pattern_idx * max_num_panels + panel_id])
                correct_num_edges += panel_correct

                if pattern_names is not None and not panel_correct:  # pattern len predicted wrongly
                    print('NumbersInPanelsAccuracies::{}::panel {}:: {} edges instead of {}'.format(
                        pattern_names[pattern_idx], panel_id,
                        predicted_num_edges, gt_num_edges[pattern_idx * max_num_panels + panel_id]))
    
            # update num panels stats
            correct_len = (predicted_num_panels == gt_panel_nums[pattern_idx])
            correct_pattern_mask[pattern_idx] = correct_len
            correct_num_panels += correct_len

            if pattern_names is not None and not correct_len:  # pattern len predicted wrongly
                print('NumbersInPanelsAccuracies::{}::{} panels instead of {}'.format(
                    pattern_names[pattern_idx], predicted_num_panels, gt_panel_nums[pattern_idx]))

            # update num edges stats (averaged per panel)
            num_edges_accuracies += correct_num_edges / gt_panel_nums[pattern_idx]
            if correct_len:
                num_edges_in_correct += correct_num_edges / gt_panel_nums[pattern_idx]
        
        # average by batch
        return (
            correct_num_panels / batch_size, 
            num_edges_accuracies / batch_size, 
            correct_pattern_mask,   # which patterns in a batch have correct number of panels? 
            num_edges_in_correct / correct_pattern_mask.sum()  # edges for correct patterns
        )
    

class PanelVertsL2():
    """
        Aims to evaluate the quality of panel shape prediction independently from loss evaluation
        * Convers panels edge lists to vertex representation (including curvature coordinates)
        * and evaluated MSE on them
    """
    def __init__(self, max_edges_in_panel, data_stats={}):
        """Info for evaluating padding vector if data statistical info is applied to it.
            * if standardization/normalization transform is applied to padding, 'data_stats' should be provided
                'data_stats' format: {'shift': <torch.tenzor>, 'scale': <torch.tensor>} 
        """
        self.data_stats = {
            'shift': torch.tensor(data_stats['shift']),
            'scale': torch.tensor(data_stats['scale']),
        }
        self.max_panel_len = max_edges_in_panel
        self.empty_panel_template = torch.zeros((max_edges_in_panel, len(self.data_stats['shift'])))
    
    def __call__(self, predicted_outlines, gt_outlines, gt_num_edges, correct_mask=None):
        """
            Evaluate on the batch of panel outlines predictoins 
            * per_panel_leading_edges -- specifies where is the start of the edge loop for GT outlines 
                that is well-matched to the predicted outlines. If not given, the default GT orientation is used
        """
        # DEBUG 

        num_panels = predicted_outlines.shape[1]

        print(predicted_outlines.shape)

        # flatten input into list of panels
        predicted_outlines = predicted_outlines.view(-1, predicted_outlines.shape[-2], predicted_outlines.shape[-1])
        gt_outlines = gt_outlines.view(-1, gt_outlines.shape[-2], gt_outlines.shape[-1])

        # devices
        if self.empty_panel_template.device != predicted_outlines.device:
            self.empty_panel_template = self.empty_panel_template.to(predicted_outlines.device)
        for key in self.data_stats:
            if self.data_stats[key].device != predicted_outlines.device:
                self.data_stats[key] = self.data_stats[key].to(predicted_outlines.device)

        # un-std
        predicted_outlines = predicted_outlines * self.data_stats['scale'] + self.data_stats['shift']
        gt_outlines = gt_outlines * self.data_stats['scale'] + self.data_stats['shift']

        # per-panel evaluation
        panel_errors = []
        correct_panel_errors = []
        # panel_mask = correct_mask.t().repeat(num_panels) if correct_mask is not None else None
        panel_mask = torch.repeat_interleave(correct_mask, num_panels) if correct_mask is not None else None
        
        # DEBUG 
        print(correct_mask.shape, panel_mask.shape)
        print(correct_mask, panel_mask)

        # panel_mask = panel_mask.view(-1) if correct_mask is not None else None

        for panel_idx in range(len(predicted_outlines)):
            prediced_panel = predicted_outlines[panel_idx]
            gt_panel = gt_outlines[panel_idx]

            # unpad both panels using correct gt info -- for simplicity of comparison
            num_edges = gt_num_edges[panel_idx]
            if num_edges < 3:  # empty panel -- skip comparison
                continue
            prediced_panel = prediced_panel[:num_edges, :]  
            gt_panel = gt_panel[:num_edges, :]

            # average squred error per vertex (not per coordinate!!) hence internal sum
            panel_errors.append(
                torch.mean(torch.sqrt(((self._to_verts(gt_panel) - self._to_verts(prediced_panel)) ** 2).sum(dim=1)))
            )

            if panel_mask is not None and panel_mask[panel_idx]:
                correct_panel_errors.append(panel_errors[-1])
        
        # mean of errors per panel
        if panel_mask is not None and len(correct_panel_errors):
            return sum(panel_errors) / len(panel_errors), sum(correct_panel_errors) / len(correct_panel_errors)
        else:
            return sum(panel_errors) / len(panel_errors), None

    def _to_verts(self, panel_edges):
        """Convert normalized panel edges into the vertex representation"""

        vert_list = [torch.tensor([0, 0]).to(panel_edges.device)]  # always starts at zero
        # edge: first two elements are the 2D vector coordinates, next two elements are curvature coordinates
        for edge in panel_edges:
            next_vertex = vert_list[-1] + edge[:2]
            edge_perp = torch.tensor([-edge[1], edge[0]]).to(panel_edges.device)

            # NOTE: on non-curvy edges, the curvature vertex in panel space will be on the previous vertex
            #       it might result in some error amplification, but we could not find optimal yet simple solution
            next_curvature = vert_list[-1] + edge[2] * edge[:2]  # X curvature coordinate
            next_curvature = next_curvature + edge[3] * edge_perp  # Y curvature coordinate

            vert_list.append(next_curvature)
            vert_list.append(next_vertex)

        vertices = torch.stack(vert_list)

        # align with the center
        vertices = vertices - torch.mean(vertices, axis=0)  # shift to average coordinate

        return vertices


class UniversalL2():
    """
        Evaluate L2 on the provided (un-standardized) data -- useful for 3D placement
    """
    def __init__(self, data_stats={}):
        """Info for un-doing the shift&scale of the data 
        """
        self.data_stats = {
            'shift': torch.tensor(data_stats['shift']),
            'scale': torch.tensor(data_stats['scale']),
        }
    
    def __call__(self, predicted, gt, correct_mask=None):
        """
         Evaluate on the batch of predictions 
         Used for rotation/translation evaluations which have input shape
         (#batch, #panels, #feature)
        """
        num_panels = predicted.shape[1]
        correct_mask = torch.repeat_interleave(correct_mask, num_panels) if correct_mask is not None else None

        # flatten input 
        predicted = predicted.view(-1, predicted.shape[-1])
        gt = gt.view(-1, gt.shape[-1])

        # devices
        for key in self.data_stats:
            if self.data_stats[key].device != predicted.device:
                self.data_stats[key] = self.data_stats[key].to(predicted.device)

        # un-std
        predicted = predicted * self.data_stats['scale'] + self.data_stats['shift']
        gt = gt * self.data_stats['scale'] + self.data_stats['shift']

        L2_norms = torch.sqrt(((gt - predicted) ** 2).sum(dim=1))

        if correct_mask is not None and len(gt[correct_mask]):
            correct_L2_norms = torch.mean(torch.sqrt(((gt[correct_mask] - predicted[correct_mask]) ** 2).sum(dim=1)))
        else:
            correct_L2_norms = None

        return torch.mean(L2_norms), correct_L2_norms


if __name__ == "__main__":
    # debug

    stitch_eval = PatternStitchPrecisionRecall()

    tags = torch.FloatTensor(
        [[
            [
                [0, 0, 0],
                [1.2, 3., 0],
                [0, 0, 0]
            ],
            [
                [0, 3., 0],
                [0, 0, 0],
                [1.2, 3., 0],
            ]
        ]]
    )
    stitches = torch.IntTensor([
        [
            [1, 5]
        ]
    ]).transpose(0, 1)

    print(stitch_eval(tags, stitches))
