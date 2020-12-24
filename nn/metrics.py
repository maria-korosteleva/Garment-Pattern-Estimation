"""List of metrics to evalute on a model and a dataset"""

import torch
import torch.nn as nn
from data import Garment3DPatternFullDataset as PatternDataset


# ------- utils ---------
def eval_pad_vector(data_stats={}):
    # prepare padding vector used for panel padding 
    if data_stats:
        shift = torch.Tensor(data_stats['shift'])
        scale = torch.Tensor(data_stats['scale'])
        return shift / scale
    else:
        return None


# ------- custom metrics --------
class PanelLoopLoss():
    """Evaluate loss for the panel edge sequence representation property: 
        ensuring edges within panel loop & return to origin"""
    def __init__(self, data_stats={}):
        """Info for evaluating padding vector if data statistical info is applied to it.
            * if standardization/normalization transform is applied to padding, 'data_stats' should be provided
                'data_stats' format: {'shift': <torch.tenzor>, 'scale': <torch.tensor>} 
        """
        self.pad_tenzor = eval_pad_vector(data_stats)
            
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
        else:
            if self.pad_tenzor is None:  # comaring everything below with zero vector
                self.pad_tenzor = torch.zeros(original_panels.shape[-1])
        self.pad_tenzor = self.pad_tenzor.to(device=predicted_panels.device) 
            
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
            # should be close to sum of the equvalent number of pading values (since all of coords are shifted due to normalization\standardization)
            # (in case of panels, padding for edge coords should be zero, but I'm using a more generic solution here JIC)
            panel_coords_sum[el_id] = (predicted_panels[el_id][:seq_len, :2] - self.pad_tenzor[:2]).sum(axis=0)

        panel_square_sums = panel_coords_sum ** 2  # per sum square

        # batch mean of squared norms of per-panel final points:
        return panel_square_sums.sum() / (panel_square_sums.shape[0] * panel_square_sums.shape[1])


class PatternStitchLoss():
    """Evalute the quality of stitching tags provided for every edge of a pattern:
        * Free edges have tags close to zero
        * Edges connected by a stitch have the same tag
        * Edges belonging to different stitches have different tags
    """
    def __init__(self, triplet_margin=0.1, use_hardnet=True):
        self.triplet_margin = triplet_margin
        
        self.neg_loss = self.HardNet_neg_loss if use_hardnet else self.extended_triplet_neg_loss

    def __call__(self, stitch_tags, gt_stitches, gt_stitches_nums):
        """
        * stitch_tags contain tags for every panel in every pattern in the batch
        * gt_stitches contains the list of edge pairs that are stitches together.
            * with every edge indicated as (panel_id, edge_id) 
        """
        gt_stitches = gt_stitches.long()
        batch_size = stitch_tags.shape[0]
        num_stitches = gt_stitches_nums.sum()  # Ground truth number of stitches!

        flat_stitch_tags = stitch_tags.view(batch_size, -1, stitch_tags.shape[-1])  # remove panel dimention

        # https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor
        # these will have dull values due to padding in gt_stitches
        left_sides = flat_stitch_tags[torch.arange(batch_size).unsqueeze(-1), gt_stitches[:, 0, :]]
        right_sides = flat_stitch_tags[torch.arange(batch_size).unsqueeze(-1), gt_stitches[:, 1, :]]
        total_tags = torch.cat([left_sides, right_sides], dim=1)

        # tags on both sides of the stitch -- together
        similarity_loss_mat = (left_sides - right_sides) ** 2

        # Gather the loss
        similarity_loss = 0.
        for pattern_idx in range(batch_size):
            # ingore values calculated for padded part of gt_stitches 
            # average by number of stitches in pattern
            similarity_loss += (
                similarity_loss_mat[pattern_idx][:gt_stitches_nums[pattern_idx], :].sum() 
                / gt_stitches_nums[pattern_idx])

        similarity_loss /= batch_size  # average similarity by stitch

        # Push tags away from each other
        total_neg_loss = self.neg_loss(total_tags, gt_stitches_nums)
               
        # final sum
        fin_stitch_losses = similarity_loss + total_neg_loss
        stitch_loss_dict = dict(
            stitch_similarity_loss=similarity_loss,
            stitch_neg_loss=total_neg_loss
        )

        return fin_stitch_losses, stitch_loss_dict

    def extended_triplet_neg_loss(self, total_tags, gt_stitches_nums):
        """Pushes stitch tags for different stitches away from each other
            * Is based on Triplet loss formula to make the distance between tags larger than margin
            * Evaluated the loss for every tag agaist every other tag (exept for the edges that are part of the same stitch thus have to have same tags)
        """
        total_neg_loss = []
        for idx, pattern_tags in enumerate(total_tags):  # per pattern in batch
            # slice pattern tags to remove consideration for stitch padding
            half_size = len(pattern_tags) // 2
            num_stitches = gt_stitches_nums[idx]

            pattern_tags = torch.cat([
                pattern_tags[:num_stitches, :], 
                pattern_tags[half_size:half_size + num_stitches, :]])

            # eval loss
            for tag_id, tag in enumerate(pattern_tags):
                # Evaluate distance to other tags
                neg_loss = (tag - pattern_tags) ** 2

                # compare with margin
                neg_loss = self.triplet_margin - neg_loss.sum(dim=-1)  # single value per other tag

                # zero out losses for entries that should be equal to current tag
                neg_loss[tag_id] = 0  # torch.zeros_like(neg_loss[tag_id]).to(neg_loss.device)
                brother_id = tag_id + num_stitches if tag_id < num_stitches else tag_id - num_stitches
                neg_loss[brother_id] = 0  # torch.zeros_like(neg_loss[tag_id]).to(neg_loss.device)

                # ignore elements far enough from current tag
                neg_loss = torch.max(neg_loss, torch.zeros_like(neg_loss))

                # fin total
                total_neg_loss.append(neg_loss.sum() / len(neg_loss))
        # average neg loss per tag
        return sum(total_neg_loss) / len(total_neg_loss)

    def HardNet_neg_loss(self, total_tags, gt_stitches_nums):
        """Pushes stitch tags for different stitches away from each other
            * Is based on Triplet loss formula to make the distance between tags larger than margin
            * Uses trick from HardNet: only evaluate the loss on the closest negative example!
        """
        total_neg_loss = []
        for pattern_tags in total_tags:  # per pattern in batch
            # slice pattern tags to remove consideration for stitch padding
            half_size = len(pattern_tags) // 2
            num_stitches = gt_stitches_nums[idx]

            pattern_tags = torch.cat([
                pattern_tags[:num_stitches, :], 
                pattern_tags[half_size:half_size + num_stitches, :]])

            for tag_id, tag in enumerate(pattern_tags):
                # Evaluate distance to other tags
                tags_distance = ((tag - pattern_tags) ** 2).sum(dim=-1)

                # mask values corresponding to current tag for min() evaluation
                tags_distance[tag_id] = float('inf')
                brother_id = tag_id + num_stitches if tag_id < num_stitches else tag_id - num_stitches
                tags_distance[brother_id] = float('inf')

                # compare with margin
                neg_loss = self.triplet_margin - tags_distance.min()  # single value per other tag

                # ignore if all tags are far enough from current tag
                total_neg_loss.append(max(neg_loss, 0))
        # average neg loss per tag
        return sum(total_neg_loss) / len(total_neg_loss)


class PatternStitchPrecisionRecall():
    """Evaluate Precision and Recall scores for pattern stitches prediction
        NOTE: It's NOT a diffentiable evaluation
    """

    def __init__(self, data_stats=None):
        self.data_stats = data_stats
        if data_stats is not None: 
            for key in self.data_stats:
                self.data_stats[key] = torch.Tensor(self.data_stats[key])

    def __call__(self, stitch_tags, free_edge_class, gt_stitches, gt_stitches_nums, pattern_names=None):
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


class NumberOfPanelsAccuracy():
    """
        Evaluate in how many cases the number of panels in patterns were detected correctly
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

    def __call__(self, predicted_patterns, gt_panel_nums, pattern_names=None):
        """
         Evaluate on the batch of stitch tags
        """
        batch_size = predicted_patterns.shape[0]
        max_num_panels = predicted_patterns.shape[1]
        if self.empty_panel_template.device != predicted_patterns.device:
            self.empty_panel_template = self.empty_panel_template.to(predicted_patterns.device)

        correct_patterns = 0
        for pattern_idx in range(batch_size):
            # assuming all empty panels are at the end of the pattern, if any
            for back_panel_id in range(max_num_panels):
                bool_matrix = torch.isclose(
                    predicted_patterns[pattern_idx][- back_panel_id], 
                    self.empty_panel_template, atol=1.e-1)  # TODO this value might differ from what is actually used in core.py
                if not torch.all(bool_matrix):
                    break
            predicted_num_panels = max_num_panels - back_panel_id
            correct = (predicted_num_panels == gt_panel_nums[pattern_idx])
            correct_patterns += correct

            if pattern_names is not None and not correct:  # pattern len predicted wrongly
                print('NumberOfPanelsAccuracy::{}::{} panels instead of {}'.format(
                    pattern_names[pattern_idx], predicted_num_panels, gt_panel_nums[pattern_idx]))
        
        # average by batch
        return float(correct_patterns) / batch_size
    

# ------- Model evaluation shortcut -------------
def eval_metrics(model, data_wrapper, section='test'):
    """Evalutes current model on the given dataset section"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    if hasattr(model, 'with_quality_eval'):
        model.with_quality_eval = True  # force quality evaluation for models that support it

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
                full_loss, loss_dict, _ = model.loss(features, gt, names=batch['name'])  # use names for cleaner errors when needed

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
            [1, 5]
        ]
    ]).transpose(0, 1)

    print(stitch_eval(tags, stitches))
