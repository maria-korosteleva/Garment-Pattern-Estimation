"""
    List of metrics to evalute on a model and a dataset, along with pre-processing methods needed for such evaluation
"""

import torch
import torch.nn as nn

from munkres import Munkres  # solving assignemnt problem

# My modules
from data import Garment3DPatternFullDataset as PatternDataset


# ------- Model evaluation shortcut -------------
def eval_metrics(model, data_wrapper, section='test'):
    """Evalutes current model on the given dataset section"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    for batch in loader:
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


# ------- custom losses --------
class PanelLoopLoss():
    """Evaluate loss for the panel edge sequence representation property: 
        ensuring edges within panel loop & return to origin"""
    def __init__(self, max_edges_in_panel, data_stats={}):
        """Info for evaluating padding vector if data statistical info is applied to it.
            * if standardization/normalization transform is applied to padding, 'data_stats' should be provided
                'data_stats' format: {'shift': <torch.tenzor>, 'scale': <torch.tensor>} 
        """
        self.data_stats = data_stats
        self.pad_vector = eval_pad_vector(data_stats)
            
    def __call__(self, predicted_panels, gt_panel_num_edges=None):
        """Evaluate loop loss on provided predicted_panels batch.
            * 'original_panels' are used to evaluate the correct number of edges of each panel in case padding is applied.
                If 'original_panels' is not given, it is assumed that there is no padding
                If data stats are not provided at init or in this call, zero vector padding is assumed
            * data_stats can be used to update padding vector on the fly
        """
        # flatten input into list of panels
        if len(predicted_panels.shape) > 3:
            predicted_panels = predicted_panels.view(-1, predicted_panels.shape[-2], predicted_panels.shape[-1])

        # correct devices
        self.pad_vector = self.pad_vector.to(predicted_panels.device)
            
        # evaluate loss
        panel_coords_sum = torch.zeros((predicted_panels.shape[0], 2))
        panel_coords_sum = panel_coords_sum.to(device=predicted_panels.device)
        for el_id in range(predicted_panels.shape[0]):
            # if unpadded len is not given, assume no padding
            seq_len = gt_panel_num_edges[el_id] if gt_panel_num_edges is not None else predicted_panels.shape[-2]
            if seq_len < 3:
                # empty panel -- no need to force loop property
                continue

            # get per-coordinate sum of edges endpoints of each panel
            # should be close to sum of the equvalent number of pading values (since all of coords are shifted due to normalization\standardization)
            # (in case of panels, padding for edge coords should be zero, but I'm using a more generic solution here JIC)
            panel_coords_sum[el_id] = (predicted_panels[el_id][:seq_len, :2] - self.pad_vector[:2]).sum(axis=0)

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
            * with every edge indicated as (pattern_edge_id) assuming panels order is known, and panels are padded to the same size
        * per_panel_leading_edges -- specifies where is the start of the edge loop for GT outlines 
                that is well-matched to the predicted outlines. 
                If not given, current edge order (in stitch tags) is assumed to match the one used in ground truth panels
        """
        gt_stitches = gt_stitches.long()
        batch_size = stitch_tags.shape[0]
        max_num_panels = stitch_tags.shape[1]
        max_panel_len = stitch_tags.shape[-2]
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
        for idx, pattern_tags in enumerate(total_tags):  # per pattern in batch
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
                    #  -> assuming this is an empty panel
                    # skipping the rest of the panels -- assuming they are also empty
                    break
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
            correct_num_panels += correct_len

            if pattern_names is not None and not correct_len:  # pattern len predicted wrongly
                print('NumbersInPanelsAccuracies::{}::{} panels instead of {}'.format(
                    pattern_names[pattern_idx], predicted_num_panels, gt_panel_nums[pattern_idx]))

            # update num edges stats (averaged per panel)
            num_edges_accuracies += correct_num_edges / gt_panel_nums[pattern_idx]
        
        # average by batch
        return correct_num_panels / batch_size, num_edges_accuracies / batch_size
    

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
    
    def __call__(self, predicted_outlines, gt_outlines, gt_num_edges):
        """
            Evaluate on the batch of panel outlines predictoins 
            * per_panel_leading_edges -- specifies where is the start of the edge loop for GT outlines 
                that is well-matched to the predicted outlines. If not given, the default GT orientation is used
        """
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
        
        # mean of errors per panel
        return sum(panel_errors) / len(panel_errors)

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
    
    def __call__(self, predicted, gt):
        """
         Evaluate on the batch of predictions 
        """
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

        return torch.mean(L2_norms)


# ---------- Composition loss -------------

class ComposedPatternLoss():
    """
        Main (callable) class to define a loss on pattern prediction as composition of components
        NOTE: relies on the GT structure for pattern desctiption as defined in Pattern datasets 
    """
    def __init__(self, data_config, in_config={}):
        """
            Initialize loss components
            Accepts (in in_config):
            * Requested list of components
            * Additional configurations for losses (e.g. edge-origin agnostic evaluation)
            * data_stats -- for correct definition of losses
        """
        self.config = {  # defults
            'loss_components': ['shape'],  # 'loop',  
            'quality_components': [],  # 'loop',  
            'loop_loss_weight': 1.,
            'stitch_tags_margin': 0.3,
            'epoch_with_stitches': 40, 
            'stitch_supervised_weight': 0.1,   # only used when explicit stitches are enabled
            'stitch_hardnet_version': False,
            'panel_origin_invariant_loss': True,
            'panel_order_inariant_loss': True,
            'order_by': 'placement'
        }
        self.config.update(in_config)  # override with requested settings

        self.with_quality_eval = True  # quality evaluation switch -- may allow to speed up the loss evaluation if False

        # Convenience properties
        self.l_components = self.config['loss_components']
        self.q_components = self.config['quality_components'] 

        self.max_panel_len = data_config['max_panel_len']
        self.max_pattern_size = data_config['max_pattern_len']

        data_stats = data_config['standardize']
        self.gt_outline_stats = {
            'shift': data_stats['gt_shift']['outlines'], 
            'scale': data_stats['gt_scale']['outlines']
        }

        #  ----- Defining loss objects --------
        # NOTE I have to make a lot of 'ifs' as all losses have different function signatures
        # So, I couldn't come up with more consize defitions
        

        if 'shape' in self.l_components or 'rotation' in self.l_components or 'translation' in self.l_components:
            self.regression_loss = nn.MSELoss()  
        
        if 'loop' in self.l_components:
            self.loop_loss = PanelLoopLoss(self.max_panel_len, data_stats=self.gt_outline_stats)
        
        if 'stitch' in self.l_components:
            self.stitch_loss = PatternStitchLoss(
                self.config['stitch_tags_margin'], use_hardnet=self.config['stitch_hardnet_version'])
        
        if 'stitch_supervised' in self.l_components:
            self.stitch_loss_supervised = nn.MSELoss()

        if 'free_class' in self.l_components:
            self.free_edge_class_loss = nn.BCEWithLogitsLoss()  # binary classification loss
        
        # -------- quality metrics ------
        if 'shape' in self.q_components:
            self.pattern_shape_quality = PanelVertsL2(self.max_panel_len, data_stats=self.gt_outline_stats)

        if 'discrete' in self.q_components:
            self.pattern_nums_quality = NumbersInPanelsAccuracies(
                self.max_panel_len, data_stats=self.gt_outline_stats)

        if 'rotation' in self.q_components:
            self.rotation_quality = UniversalL2(data_stats={
                'shift': data_stats['gt_shift']['rotations'], 
                'scale': data_stats['gt_scale']['rotations']}
            )
        if 'translation' in self.q_components:
            self.translation_quality = UniversalL2(data_stats={
                'shift': data_stats['gt_shift']['translations'], 
                'scale': data_stats['gt_scale']['translations']}
            )
        if 'stitch' in self.q_components:
            self.stitch_quality = PatternStitchPrecisionRecall(
                data_stats={
                    'shift': data_stats['gt_shift']['stitch_tags'], 
                    'scale': data_stats['gt_scale']['stitch_tags']
                } if data_config['explicit_stitch_tags'] else None
            )

    def __call__(self, preds, ground_truth, names=None, epoch=1000):
        """Evalute loss when predicting patterns.
            * Predictions are expected to follow the default GT structure, 
                but don't have to have all components -- as long as provided prediction is sufficient for
                evaluation of requested losses
            * default epoch is some large value to trigger stitch evaluation
            * Function returns True in third parameter at the moment of the loss stucture update
        """
        self.device = preds['outlines'].device
        loss_dict = {}
        full_loss = 0.

        # match devices with prediction
        for key in ground_truth:
            ground_truth[key] = ground_truth[key].to(self.device)  

        # ------ GT pre-processing --------
        if self.config['panel_order_inariant_loss']:  # match panel order
            gt_rotated = self._gt_order_match(preds, ground_truth, epoch) 
        else:  # keep original
            gt_rotated = ground_truth
        
        gt_num_edges = gt_rotated['num_edges'].int().view(-1)  # flatten

        if self.config['panel_origin_invariant_loss']:  # for origin-agnistic loss evaluation
            gt_rotated = self._rotate_gt(preds, gt_rotated, gt_num_edges, epoch)

        # ---- Losses ------
        main_losses, main_dict = self._main_losses(preds, gt_rotated, gt_num_edges)
        full_loss += main_losses
        loss_dict.update(main_dict)

        # stitch losses -- conditioned on the current process in training
        if epoch >= self.config['epoch_with_stitches'] and (
                'stitch' in self.l_components
                or 'stitch_supervised' in self.l_components
                or 'free_class' in self.l_components):
            losses, stitch_loss_dict = self._stitch_losses(preds, gt_rotated, gt_num_edges)
            full_loss += losses
            loss_dict.update(stitch_loss_dict)


        # ---- Quality metrics  ----
        if self.with_quality_eval:
            with torch.no_grad():
                quality_breakdown = self._main_quality_metrics(preds, gt_rotated, gt_num_edges, names)
                loss_dict.update(quality_breakdown)

                # stitches quality
                if epoch >= self.config['epoch_with_stitches']:
                    quality_breakdown = self._stitch_quality_metrics(preds, gt_rotated, gt_num_edges, names)
                    loss_dict.update(quality_breakdown)

        # final loss; breakdown for analysis; indication if the loss structure has changed on this evaluation
        return full_loss, loss_dict, epoch == self.config['epoch_with_stitches']

    # ------- evaluation breakdown -------
    def _main_losses(self, preds, ground_truth, gt_num_edges):
        """
            Main loss components. Evaluated in the same way regardless of the training stage
        """
        full_loss = 0.
        loss_dict = {}

        if 'shape' in self.l_components:
            pattern_loss = self.regression_loss(preds['outlines'], ground_truth['outlines'])
            full_loss += pattern_loss
            loss_dict.update(pattern_loss=pattern_loss)
            
        if 'loop' in self.l_components:
            loop_loss = self.loop_loss(preds['outlines'], gt_num_edges)
            full_loss += self.config['loop_loss_weight'] * loop_loss
            loss_dict.update(loop_loss=loop_loss)
            
        if 'rotation' in self.l_components:
            # independent from panel loop origin by design
            rot_loss = self.regression_loss(preds['rotations'], ground_truth['rotations'])
            full_loss += rot_loss
            loss_dict.update(rotation_loss=rot_loss)
        
        if 'translation' in self.l_components:
            # independent from panel loop origin by design
            translation_loss = self.regression_loss(preds['translations'], ground_truth['translations'])
            full_loss += translation_loss
            loss_dict.update(translation_loss=translation_loss)

        return full_loss, loss_dict

    def _stitch_losses(self, preds, ground_truth, gt_num_edges):
        """
            Evaluate losses related to stitch info. Maybe calles or not depending on the training stage
        """
        full_loss = 0.
        loss_dict = {}

        if 'stitch' in self.l_components: 
            # Pushing stitch tags of the stitched edges together, and apart from all the other stitch tags
            stitch_loss, stitch_loss_breakdown = self.stitch_loss(
                preds['stitch_tags'], ground_truth['stitches'], ground_truth['num_stitches'])
            loss_dict.update(stitch_loss_breakdown)
            full_loss += stitch_loss
        
        if 'stitch_supervised' in self.l_components:
            stitch_sup_loss = self.stitch_loss_supervised(
                preds['stitch_tags'], ground_truth['stitch_tags'])      
            loss_dict.update(stitch_supervised_loss=stitch_sup_loss)
            full_loss += self.config['stitch_supervised_weight'] * stitch_sup_loss

        if 'free_class' in self.l_components:
            # free\stitches edges classification
            free_edges_loss = self.free_edge_class_loss(
                preds['free_edges_mask'], ground_truth['free_edges_mask'].type(torch.FloatTensor).to(self.device))
            loss_dict.update(free_edges_loss=free_edges_loss)
            full_loss += free_edges_loss

        return full_loss, loss_dict

    def _main_quality_metrics(self, preds, ground_truth, gt_num_edges, names):
        """
            Evaluate quality components -- these are evaluated in the same way regardless of the training stage
        """
        loss_dict = {}

        if 'shape' in self.q_components:
            shape_l2 = self.pattern_shape_quality(
                preds['outlines'], ground_truth['outlines'], gt_num_edges)
            loss_dict.update(panel_shape_l2=shape_l2)

        if 'discrete' in self.q_components:
            num_panels_acc, num_edges_acc = self.pattern_nums_quality(
                preds['outlines'], gt_num_edges, ground_truth['num_panels'], pattern_names=names)
            loss_dict.update(num_panels_accuracy=num_panels_acc, num_edges_accuracy=num_edges_acc,)
            
        if 'rotation' in self.q_components:
            rotation_l2 = self.rotation_quality(
                preds['rotations'], ground_truth['rotations'])
            loss_dict.update(rotation_l2=rotation_l2)

        if 'translation' in self.q_components:
            translation_l2 = self.translation_quality(
                preds['translations'], ground_truth['translations'])
            loss_dict.update(translation_l2=translation_l2)
    
        return loss_dict

    def _stitch_quality_metrics(self, preds, ground_truth, gt_num_edges, names):
        """
            Quality components related to stitches prediction. May be called separately from main components 
            arrording to the training stage
        """
        loss_dict = {}
        if 'stitch' in self.q_components:
            stitch_prec, stitch_recall = self.stitch_quality(
                preds['stitch_tags'], preds['free_edges_mask'], 
                ground_truth['stitches'].type(torch.IntTensor).to(self.device), 
                ground_truth['num_stitches'],
                pattern_names=names)
            loss_dict.update(stitch_precision=stitch_prec, stitch_recall=stitch_recall)
        
        if 'free_class' in self.q_components:
            free_class = torch.round(torch.sigmoid(preds['free_edges_mask']))
            gt_mask = ground_truth['free_edges_mask'].to(preds['free_edges_mask'].device)
            acc = (free_class == gt_mask).sum().float() / gt_mask.numel()

            loss_dict.update(free_edge_acc=acc)

        return loss_dict

    # ------ Ground truth panel order match -----
    def _gt_order_match(self, preds, ground_truth, epoch):
        """
            Find the permutation of panel in GT that is best matched with the prediction (by geometry)
            and return the GT object with all properties updated according to this permutation 
        """
        with torch.no_grad():
            gt_updated = {}

            # Match the order
            if self.config['order_by'] == 'placement':
                if 'translations' not in preds or 'rotations' not in preds:
                    raise ValueError('ComposedPatternLoss::Error::Ordering by placement requested but placement is not predicted')

                pred_placement = torch.cat([preds['translations'], preds['rotations']], dim=-1)
                gt_placement = torch.cat([ground_truth['translations'], ground_truth['rotations']], dim=-1)

                gt_permutation = self._panel_order_match(pred_placement, gt_placement, ground_truth['num_panels'])
            else:
                raise NotImplemented('ComposedPatternLoss::Error::Ordering by requested feature <{}> is not implemented'.format(
                    self.config['order_by']
                ))

            # Update gt info according to the permutation
            gt_updated['outlines'] = self._feature_permute(
                ground_truth['outlines'], gt_permutation, ground_truth['num_panels'])

            gt_updated['num_edges'] = self._feature_permute(
                ground_truth['num_edges'], gt_permutation, ground_truth['num_panels'])

            # TODO update the info according to chosen leading edges

            if 'rotation' in self.l_components:
                gt_updated['rotations'] = self._feature_permute(
                    ground_truth['rotations'], gt_permutation, ground_truth['num_panels'])
            if 'translation' in self.l_components:
                gt_updated['translations'] = self._feature_permute(
                    ground_truth['translations'], gt_permutation, ground_truth['num_panels'])
            
            if epoch >= self.config['epoch_with_stitches'] and (
                    'stitch' in self.l_components
                    or 'stitch_supervised' in self.l_components
                    or 'free_class' in self.l_components):  # if there is any stitch-related evaluation

                gt_updated['stitches'] = self._stitch_after_permute( 
                    ground_truth['stitches'], ground_truth['num_stitches'], 
                    gt_permutation, self.max_panel_len
                )
                gt_updated['free_edges_mask'] = self._feature_permute(
                    ground_truth['free_edges_mask'], gt_permutation, ground_truth['num_panels'])
                
                if 'stitch_supervised' in self.l_components:
                    gt_updated['stitch_tags'] = self._feature_permute(
                        ground_truth['stitch_tags'], gt_permutation, ground_truth['num_panels'])

            # keep the references to the rest of the gt data as is
            for key in ground_truth:
                if key not in gt_updated:
                    gt_updated[key] = ground_truth[key]

        return gt_updated

    def _panel_order_match(self, pred_features, gt_features, num_panels):
        """
            Find the best-matching permutation of gt panels to the predicted panels (in panel order)
            based on the provided panel features
        """
        with torch.no_grad():
            assignment_solver = Munkres()  # one solver for all problems

            per_pettern_permutation = []
            # per-pattern processing
            for pattern_idx in range(pred_features.shape[0]):
                gt_len = num_panels[pattern_idx]

                # distances between panels
                dist_matrix = torch.cdist(
                    pred_features[pattern_idx][:gt_len].view(gt_len, -1), 
                    gt_features[pattern_idx][:gt_len].view(gt_len, -1))

                # find optimal order assignment, see https://pypi.org/project/munkres/1.0.9/
                indexes = assignment_solver.compute(dist_matrix)
                if len(indexes) != gt_len:
                    raise RuntimeError("ComposedPatternLoss::Error:: Failed to match panel order" )

                # Gather the GT in requested order
                match = [-1] * gt_len
                for left, right in indexes:
                    match[left] = right

                per_pettern_permutation.append(match)
        
        return per_pettern_permutation

    def _panel_order_match_shape(self, predicted_patterns, gt_patterns, num_panels, gt_num_edges):
        """
            Find the best-matching permutation of gt panels to the predicted panels (in panel order)
        """
        with torch.no_grad():
            assignment_solver = Munkres()  # one solver for all problems

            chosen_patterns = []
            per_pettern_permutation = []
            all_panel_leading_edges = []
            # per-pattern processing
            for pattern_idx in range(predicted_patterns.shape[0]):
                gt_len = num_panels[pattern_idx]

                # distances between panels
                if self.config['panel_origin_invariant_loss']:

                    print('Order + Origin')

                    # distance between panels is based on the best rotation of GT panel
                    dist_matrix = torch.empty((gt_len, gt_len)).to(predicted_patterns.device)
                    panel_leading_edges_pairs = torch.empty_like(dist_matrix).to(predicted_patterns.device)
                    chosen_panels = []  # list but cornains chosen order for all panels
                    # calculate the optimal distance for the every pair panel-panel
                    for pred_panel_id in range(gt_len):
                        for gt_panel_id in range(gt_len):
                            # optimal match
                            gt_chosen, leading_edge, dist = self._panel_egde_match(
                                predicted_patterns[pattern_idx][pred_panel_id], 
                                gt_patterns[pattern_idx][gt_panel_id], 
                                gt_num_edges[pattern_idx][gt_panel_id])
                            
                            dist_matrix[pred_panel_id][gt_panel_id] = dist
                            chosen_panels.append(gt_chosen)
                            panel_leading_edges_pairs[pred_panel_id][gt_panel_id] = leading_edge

                else:
                    dist_matrix = torch.cdist(
                        predicted_patterns[pattern_idx][:gt_len].view(gt_len, -1), 
                        gt_patterns[pattern_idx][:gt_len].view(gt_len, -1))

                # find optimal order assignment, see https://pypi.org/project/munkres/1.0.9/
                indexes = assignment_solver.compute(dist_matrix)
                if len(indexes) != gt_len:
                    raise RuntimeError("ComposedPatternLoss::Error:: Failed to match panel order" )

                # Gather the GT in requested order
                match = [-1] * gt_len
                for left, right in indexes:
                    match[left] = right

                if self.config['panel_origin_invariant_loss']:
                    # Pass on the chosen leading edges with account for padded panels
                    all_panel_leading_edges += [panel_leading_edges_pairs[i][match[i]] for i in range(gt_len)] + [0] * (gt_patterns.shape[1] - gt_len)

                    # updated GT based on origin matching choice
                    # flat indexing as we use the list as storage for matched panels
                    rearranged_pattern = torch.stack([chosen_panels[i * gt_len + match[i]] for i in range(gt_len)]).to(predicted_patterns.device)
                else:
                    rearranged_pattern = torch.stack([gt_patterns[pattern_idx][i] for i in match]).to(predicted_patterns.device)

                if gt_len < gt_patterns.shape[1]:
                    rearranged_pattern = torch.cat([rearranged_pattern, gt_patterns[pattern_idx][gt_len: ]])

                chosen_patterns.append(rearranged_pattern)
                per_pettern_permutation.append(match)
        
        return torch.stack(chosen_patterns).to(predicted_patterns.device), per_pettern_permutation, all_panel_leading_edges if len(all_panel_leading_edges) else None

    @staticmethod
    def _feature_permute(pattern_features, permutation, num_panels):
        """
            Permute all given features (in the batch) according to given panel order permutation
        """
        with torch.no_grad():
            all_updated = []
            for pattern_idx in range(len(pattern_features)):
                updated_feature = torch.stack([pattern_features[pattern_idx][i] for i in permutation[pattern_idx]])
                # return padding too
                if num_panels[pattern_idx] < pattern_features.shape[1]:  # less them max number of panels
                    updated_feature = torch.cat([
                        updated_feature, 
                        pattern_features[pattern_idx][num_panels[pattern_idx]:]
                    ])
                all_updated.append(updated_feature)
        
        return torch.stack(all_updated).to(pattern_features.device)

    @staticmethod
    def _stitch_after_permute(stitches, stitches_num, permutation, max_panel_len):
        """
            Update edges ids in stitch info after panel order permutation
        """
        with torch.no_grad():  # GT updates don't require gradient compute
            # add pattern dimention
            for pattern_id in range(len(stitches)):
                # re-assign GT edge ids according to shift
                for side in (0, 1):
                    for i in range(stitches_num[pattern_id]):
                        edge_id = stitches[pattern_id][side][i]
                        panel_id = edge_id // max_panel_len
                        in_panel_edge_id = edge_id - (panel_id * max_panel_len)

                        # where is this panel placed
                        new_panel_id = permutation[pattern_id].index(panel_id)  

                        # update with pattern-level edge id
                        stitches[pattern_id][side][i] = new_panel_id * max_panel_len + in_panel_edge_id
                
        return stitches

    # ------ Ground truth panel shift  ---------
    def _rotate_gt(self, preds, ground_truth, gt_num_edges, epoch):
        """
            Create a new GT object where panels are rotated to best match the predicted panels
        """
        with torch.no_grad():
            gt_updated = {}
            # for origin-agnistic loss evaluation
            gt_updated['outlines'], panel_leading_edges = self._batch_edge_order_match(
                preds['outlines'], ground_truth['outlines'], gt_num_edges)

            if epoch >= self.config['epoch_with_stitches'] and (
                    'stitch' in self.l_components
                    or 'stitch_supervised' in self.l_components
                    or 'free_class' in self.l_components):  # if there is any stitch-related evaluation
                gt_updated['stitches'] = self._gt_stitches_shift(
                    ground_truth['stitches'], ground_truth['num_stitches'], 
                    panel_leading_edges, gt_num_edges,
                    self.max_pattern_size, self.max_panel_len
                )
                gt_updated['free_edges_mask'] = self._per_panel_shift(
                    ground_truth['free_edges_mask'], 
                    panel_leading_edges, gt_num_edges)
                
                if 'stitch_supervised' in self.l_components:
                    gt_updated['stitch_tags'] = self._per_panel_shift(
                        ground_truth['stitch_tags'], panel_leading_edges, gt_num_edges)
            
            # keep the references to the rest of the gt data as is
            for key in ground_truth:
                if key not in gt_updated:
                    gt_updated[key] = ground_truth[key]

        return gt_updated

    @staticmethod
    def _batch_edge_order_match(predicted_panels, gt_panels, gt_num_edges):
        """
            Try different first edges of GT panels to find the one best matching with prediction
        """
        batch_size = predicted_panels.shape[0]
        if len(predicted_panels.shape) > 3:
            predicted_panels = predicted_panels.view(-1, predicted_panels.shape[-2], predicted_panels.shape[-1])
        if gt_panels is not None and len(gt_panels.shape) > 3:
            gt_panels = gt_panels.view(-1, gt_panels.shape[-2], gt_panels.shape[-1])
        
        chosen_panels = []
        leading_edges = []
        # choose the closest version of original panel for each predicted panel
        with torch.no_grad():
            for el_id in range(predicted_panels.shape[0]):
                num_edges = gt_num_edges[el_id]

                # Find loop origin with min distance to predicted panel
                chosen_panel, leading_edge, _ = ComposedPatternLoss._panel_egde_match(
                    predicted_panels[el_id], gt_panels[el_id], num_edges)

                # update choice
                chosen_panels.append(chosen_panel)
                leading_edges.append(leading_edge)

        chosen_panels = torch.stack(chosen_panels).to(predicted_panels.device)

        # reshape into pattern batch
        return chosen_panels.view(batch_size, -1, gt_panels.shape[-2], gt_panels.shape[-1]), leading_edges

    @staticmethod
    def _panel_egde_match(pred_panel, gt_panel, num_edges):
        """
            Find the optimal origin for gt panel that matches with the pred_panel best
        """
        # TODO Faster version? -- I think I already did smth like with stitch tags

        shifted_gt_panel = gt_panel
        min_dist = ((pred_panel - shifted_gt_panel) ** 2).sum()
        chosen_panel = shifted_gt_panel
        leading_edge = 0
        for i in range(1, num_edges):  # will skip comparison if num_edges is 0 -- empty panels
            shifted_gt_panel = ComposedPatternLoss._rotate_edges(shifted_gt_panel, num_edges)
            dist = ((pred_panel - shifted_gt_panel) ** 2).sum()
            if dist < min_dist:
                min_dist = dist
                chosen_panel = shifted_gt_panel
                leading_edge = i
        
        return chosen_panel, leading_edge, min_dist

    @staticmethod
    def _per_panel_shift(panel_features, per_panel_leading_edges, panel_num_edges):
        """
            Shift given panel features accorging to the new edge loop orientations given
        """
        pattern_size = panel_features.shape[1]
        with torch.no_grad():
            for pattern_idx in range(len(panel_features)):
                for panel_idx in range(pattern_size):
                    edge_id = per_panel_leading_edges[pattern_idx * pattern_size + panel_idx] 
                    num_edges = panel_num_edges[pattern_idx * pattern_size + panel_idx]       
                    if num_edges < 3:  # just skip empty panels
                        continue
                    if edge_id:  # not zero -- shift needed. For empty panels its always zero
                        current_panel = panel_features[pattern_idx][panel_idx]
                        # requested edge goes into the first place
                        # padded area is left in place
                        panel_features[pattern_idx][panel_idx] = torch.cat(
                            (current_panel[edge_id:num_edges], current_panel[: edge_id], current_panel[num_edges:]))
        return panel_features

    @staticmethod
    def _gt_stitches_shift(
            gt_stitches, gt_stitches_nums, 
            per_panel_leading_edges, 
            gt_num_edges,
            max_num_panels, max_panel_len):
        """
            Re-number the edges in ground truth according to the perdiction-gt edges mapping indicated in per_panel_leading_edges
        """
        with torch.no_grad():  # GT updates don't require gradient compute
            # add pattern dimention
            # TODO less nested loops!!!!
            for pattern_id in range(len(gt_stitches)):
                # re-assign GT edge ids according to shift
                for side in (0, 1):
                    for i in range(gt_stitches_nums[pattern_id]):
                        edge_id = gt_stitches[pattern_id][side][i]
                        panel_id = edge_id // max_panel_len
                        global_panel_id = pattern_id * max_num_panels + panel_id  # panel id in the batch
                        new_ledge = per_panel_leading_edges[global_panel_id]
                        panel_num_edges = gt_num_edges[global_panel_id]  # never references to empty (padding) panel->always positive number

                        inner_panel_id = edge_id - (panel_id * max_panel_len)  # edge id within panel
                        
                        # shift edge within panel
                        new_in_panel_id = inner_panel_id - new_ledge if inner_panel_id >= new_ledge else (
                            panel_num_edges - (new_ledge - inner_panel_id))
                        # update with pattern-level edge id
                        gt_stitches[pattern_id][side][i] = panel_id * max_panel_len + new_in_panel_id
                
        return gt_stitches

    @staticmethod
    def _rotate_edges(panel, num_edges):
        """
            Rotate the start of the loop to the next edge
        """
        panel = torch.cat((panel[1:num_edges], panel[0:1, :], panel[num_edges:]))

        return panel


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
