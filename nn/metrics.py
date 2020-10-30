"""List of metrics to evalute on a model and a dataset"""

import torch
import torch.nn as nn
from data import Garment3DPatternFullDataset as PatternDataset


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
        
        
        # prepare for padding comparison
        with_unpadding = original_panels is not None and original_panels.nelement() > 0  # existing non-empty tensor
        if with_unpadding:
            # flatten if not already 
            if len(original_panels.shape) > 3:
                original_panels = original_panels.view(-1, original_panels.shape[-2], original_panels.shape[-1])
            if data_stats:  # update pad vector
                self._eval_pad_vector(data_stats)
            if self.pad_tenzor is None:  # still not defined -> assume zero vector for padding
                self.pad_tenzor = torch.zeros(original_panels.shape[-1])
            pad_tenzor_propagated = self.pad_tenzor.repeat(original_panels.shape[1], 1)
            pad_tenzor_propagated = pad_tenzor_propagated.to(device=predicted_panels.device)
            
        # evaluate loss
        panel_coords_sum = torch.zeros((predicted_panels.shape[0], 2))
        panel_coords_sum = panel_coords_sum.to(device=predicted_panels.device)
        for el_id in range(predicted_panels.shape[0]):
            if with_unpadding:
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
            shift = torch.Tensor(data_stats['shift'])
            scale = torch.Tensor(data_stats['scale'])
            self.pad_tenzor = - shift / scale
        else:
            self.pad_tenzor = None


class PatternStitchLoss():
    """Evalute the quality of stitching tags provided for every edge of a pattern:
        * Free edges have tags close to zero
        * Edges connected by a stitch have the same tag
        * Edges belonging to different stitches have 
    """
    def __init__(self, triplet_margin=0.1):
        self.triplet_margin = triplet_margin

    def __call__(self, stitch_tags, gt_stitches):
        """
        * stitch_tags contain tags for every panel in every pattern in the batch
        * gt_stitches contains the list of edge pairs that are stitches together.
            * with every edge indicated as (panel_id, edge_id) 
        """
        stitch_losses = []
        free_edge_losses = []
        tag_len = stitch_tags.shape[-1]
        for pattern_idx in range(stitch_tags.shape[0]):
            pattern = stitch_tags[pattern_idx]
    
            # build up losses for every stitch
            for stitch_id in range(gt_stitches[pattern_idx].shape[0]):
                # same stitch -- same tags
                stitch = gt_stitches[pattern_idx][stitch_id]
                similarity_loss = (pattern[stitch[0][0]][stitch[0][1]] - pattern[stitch[1][0]][stitch[1][1]]) ** 2
                similarity_loss = similarity_loss.sum() / tag_len  # average

                neg_losses = []
                # different stitches -- different tags
                for other_id in range(gt_stitches[pattern_idx].shape[0]):
                    if stitch_id != other_id:
                        other_stitch = gt_stitches[pattern_idx][other_id]
                        for side_1 in [0, 1]:
                            for side_2 in [0, 1]:
                                neg_loss = (pattern[stitch[side_1][0]][stitch[side_1][1]] - pattern[other_stitch[side_2][0]][other_stitch[side_2][1]]) ** 2
                                neg_losses.append(max(self.triplet_margin - neg_loss.sum() / tag_len, 0))  # ensure minimal distanse on average
                # Compare to zero too (both sides)
                neg_losses.append(max(self.triplet_margin - (pattern[stitch[0][0]][stitch[0][1]] ** 2).sum() / tag_len, 0))
                neg_losses.append(max(self.triplet_margin - (pattern[stitch[1][0]][stitch[1][1]] ** 2).sum() / tag_len, 0))

                # neg losses normalized by quantity
                stitch_losses.append(similarity_loss + sum(neg_losses) / len(neg_losses))
                
            # Find out which edges are not connected to anything
            connectivity_mat = torch.zeros((pattern.shape[0], pattern.shape[1]), dtype=torch.bool)
            for stitch in gt_stitches[pattern_idx]:
                connectivity_mat[stitch[0][0]][stitch[0][1]] = True
                connectivity_mat[stitch[1][0]][stitch[1][1]] = True

            for panel_id in range(connectivity_mat.shape[0]):
                for edge_id in range(connectivity_mat.shape[1]):
                    if not connectivity_mat[panel_id][edge_id]:
                        # free edge to have zero tags
                        free_edge_losses.append(pattern[panel_id][edge_id] ** 2 / tag_len)  # average error
            
        # batch mean losses
        fin_stitch_losses = sum(stitch_losses) / len(stitch_losses)
        fin_free_losses = torch.cat(free_edge_losses).sum() / len(free_edge_losses)  # average per edge

        return fin_stitch_losses, fin_free_losses


class PatternStitchPrecisionRecall():
    """Evaluate Precision and Recall scores for pattern stitches prediction
        NOTE: It's NOT a diffentiable evaluation
    """

    def __init__(self):
        pass

    def __call__(self, stitch_tags, gt_stitches):
        """
         Evaluate on the batch of stitch tags
        """
        tot_precision = 0
        tot_recall = 0
        for pattern_idx in range(stitch_tags.shape[0]):
            stitch_list = PatternDataset.tags_to_stitches(stitch_tags[pattern_idx])
            stitch_list = torch.IntTensor(stitch_list).to(gt_stitches.device)

            correct_stitches = 0
            # compare stitches
            for detected in stitch_list:
                for actual in gt_stitches[pattern_idx]:
                    # order-invariant comparison of stitch sides
                    if (all(detected[0] == actual[0]) and all(detected[1] == actual[1])) \
                            or (all(detected[1] == actual[0]) and all(detected[0] == actual[1])):
                        correct_stitches += 1

            # precision -- how many of the detected stitches are actually there
            tot_precision += correct_stitches / len(stitch_list) if len(stitch_list) else 0
            # recall -- how many of the actual stitches were detected
            tot_recall += correct_stitches / len(gt_stitches[pattern_idx])
        
        # evrage by batch
        return tot_precision / stitch_tags.shape[0], tot_recall / stitch_tags.shape[0]

    def on_loader(self, data_loader, model):
        """Evaluate recall&precision of stitch detection on the full data loader"""

        with torch.no_grad():
            tot_precision = tot_recall = 0
            for batch in data_loader:
                predictions = model(batch['features'])
                batch_precision, batch_recall = self(predictions['stitch_tags'], batch['ground_truth']['stitches'])
                tot_precision += batch_precision
                tot_recall += batch_recall

        return tot_precision / len(data_loader), tot_recall / len(data_loader)


# ------- Model evaluation shortcut -------------
def eval_metrics(model, data_wrapper, section='test'):
    """Evalutes current model on the given dataset section"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        current_metrics = {}
        loader = data_wrapper.get_loader(section)
        if loader:
            current_metrics = dict.fromkeys(['full_loss'], 0)
            model_defined = 0
            loop_loss = 0
            for batch in loader:
                features, gt = batch['features'].to(device), batch['ground_truth']
                if gt is None or (hasattr(gt, 'nelement') and gt.nelement() == 0):  # assume reconstruction task
                    gt = features

                # loss evaluation
                full_loss, loss_dict = model.loss(features, gt)

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
            [[0, 1], [1, 2]]
        ]
    ])

    print(stitch_eval(tags, stitches))
