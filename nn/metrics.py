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
        return (- shift / scale)
    else:
        return None


def panel_len_from_padded(padded_panel, pad_vector=None, empty_template=None):
    """
        Return length of the unpadded part of given panel (hence, number of edges)
    """
    if pad_vector is None and empty_template is None:
        return len(padded_panel)

    if empty_template is None:
        empty_template = pad_vector.repeat(padded_panel.shape[0], 1)
        empty_template = pad_tenzor_propagated.to(device=padded_panel.device)

    # unpaded length
    bool_matrix = torch.isclose(padded_panel, empty_template, atol=1.e-2)
    seq_len = (~torch.all(bool_matrix, axis=1)).sum()  # only non-padded rows

    return seq_len


# ----- for gt shifting -------
def per_panel_shift(panel_features, per_panel_leading_edges, panel_num_edges):
    """
        Shift given panel features accorging to the new edge loop orientations given
    """
    pattern_size = panel_features.shape[1]
    with torch.no_grad():
        for pattern_idx in range(len(panel_features)):
            for panel_idx in range(pattern_size):
                edge_id = per_panel_leading_edges[pattern_idx * pattern_size + panel_idx] 
                num_edges = gt_num_edges[pattern_idx * pattern_size + panel_idx]       
                if edge_id:  # not zero -- shift needed
                    panel_feature = panel_features[pattern_idx][panel_idx]
                    # requested edge goes into the first place
                    # padded area is left in place
                    panel_features[pattern_idx][panel_idx] = torch.cat(
                        (panel_features[edge_id:num_edges], panel_features[: edge_id], panel_features[num_edges:]))
    return panel_features


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
        self.max_panel_len = max_edges_in_panel
        self.pad_vector = eval_pad_vector(data_stats)
        self.empty_panel_template = self.pad_vector.repeat(self.max_panel_len, 1)
            
    def __call__(self, predicted_panels, gt_panels=None):
        """Evaluate loop loss on provided predicted_panels batch.
            * 'original_panels' are used to evaluate the correct number of edges of each panel in case padding is applied.
                If 'original_panels' is not given, it is assumed that there is no padding
                If data stats are not provided at init or in this call, zero vector padding is assumed
            * data_stats can be used to update padding vector on the fly
        """
        # flatten input into list of panels
        if len(predicted_panels.shape) > 3:
            predicted_panels = predicted_panels.view(-1, predicted_panels.shape[-2], predicted_panels.shape[-1])
        if gt_panels is not None and len(gt_panels.shape) > 3:
            gt_panels = gt_panels.view(-1, gt_panels.shape[-2], gt_panels.shape[-1])

        # correct devices
        self.empty_panel_template = self.empty_panel_template.to(predicted_panels.device)
        self.pad_vector = self.pad_vector.to(predicted_panels.device)
            
        # evaluate loss
        panel_coords_sum = torch.zeros((predicted_panels.shape[0], 2))
        panel_coords_sum = panel_coords_sum.to(device=predicted_panels.device)
        for el_id in range(predicted_panels.shape[0]):
            seq_len = panel_len_from_padded(gt_panels[el_id], empty_template=self.empty_panel_template)

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

    def __call__(self, stitch_tags, gt_stitches, gt_stitches_nums, per_panel_leading_edges=None, gt_panel_num_edges=None):
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

        if per_panel_leading_edges is not None:  # re-arrange GT edge ids 
            if gt_panel_num_edges is None:
                raise ValueError(
                    '{}::Error::need information on GT num of edges in every panel when *per_panel_leading_edges* is provided'.format(
                        self.__class__.__name__
                    )
                )
            print(gt_stitches.shape)
            gt_stitches = self.gt_stitches_shift(
                gt_stitches, gt_stitches_nums, per_panel_leading_edges, gt_panel_num_edges, max_num_panels, max_panel_len)
            print(gt_stitches.shape)

        flat_stitch_tags = stitch_tags.view(batch_size, -1, stitch_tags.shape[-1])  # remove panel dimention

        print('Final gt stitches ', gt_stitches)

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

    @staticmethod
    def gt_stitches_shift(
            gt_stitches, gt_stitches_nums, per_panel_leading_edges, 
            gt_num_edges,
            max_num_panels, max_panel_len):
        """
            Re-number the edges in ground truth according to the perdiction-gt edges mapping indicated in per_panel_leading_edges
        """
        with torch.no_grad():  # GT updates don't require gradient compute
            # add pattern dimention
            # TODO less nested loops!!!!
            print(len(per_panel_leading_edges))
            for pattern_id in range(len(gt_stitches)):
                
                print(pattern_id, gt_stitches[pattern_id])
                
                print([per_panel_leading_edges[pattern_id * max_num_panels + i] for i in range(max_num_panels)])
                # re-assign GT edge ids according to shift
                for side in (0, 1):
                    for i in range(gt_stitches_nums[pattern_id]):
                        edge_id = gt_stitches[pattern_id][side][i]
                        panel_id = edge_id // max_panel_len
                        global_panel_id = pattern_id * max_num_panels + panel_id
                        new_ledge = per_panel_leading_edges[global_panel_id]
                        panel_num_edges = gt_num_edges[global_panel_id]

                        inner_panel_id = edge_id - (panel_id * max_panel_len)
                        
                        # TODO it should be num_edges instead of max_panel_len -- right??
                        new_in_panel_id = inner_panel_id - new_ledge if inner_panel_id >= new_ledge else (
                            panel_num_edges - (new_ledge - inner_panel_id))

                        print(max_panel_len, inner_panel_id, new_ledge, new_in_panel_id)

                        gt_stitches[pattern_id][side][i] = panel_id * max_panel_len + new_in_panel_id
                
                print(gt_stitches[pattern_id])
        return gt_stitches


class PatternFreeEdgesLoss():
    """
        Evaluate the edge classification into free and non-free (connected to some other edge by a stitch)
    """
    def __init__(self):
        self.logit_loss = nn.BCEWithLogitsLoss()

    def __call__(self, pred_class, gt_free_class, per_panel_leading_edges=None, gt_panel_num_edges=None):
        """
            Evaluates the BCE loss on predicted logits
            * per_panel_leading_edges -- if provided, gt mask is re-agganged to follow the matched order of edges provided in this parameter 
        """

        if per_panel_leading_edges is not None:
            if gt_panel_num_edges is None:
                raise ValueError(
                    '{}::Error::need information on GT num of edges in every panel when *per_panel_leading_edges* is provided'.format(
                        self.__class__.__name__
                    )
                )
            # re-arrange GT labels to match the edge order
            with torch.no_grad():

                gt_free_class = gt_free_class.view(-1, gt_free_class.shape[-1])

                for panel_idx in range(len(gt_free_class)):
                    num_edges = gt_panel_num_edges[panel_idx]
                    edge_id = per_panel_leading_edges[panel_idx]
                    if edge_id:  # not zero -- shift needed
                        gt_panel = gt_free_class[panel_idx]
                        gt_free_class[panel_idx] = torch.cat((gt_panel[edge_id:num_edges], gt_panel[: edge_id], gt_panel[num_edges:]))

                gt_free_class = gt_free_class.view(pred_class.shape)


        return self.logit_loss(pred_class, gt_free_class), gt_free_class


class PanelShapeOriginAgnosticLoss(PanelLoopLoss):
    """
        Regression on Panel Shape that allows any vertex to serve as an edge loop origin in panels 
        Follow similar structure to the loop loss
    """
    def __init__(self, max_panel_len, data_stats={}):
        """Info for evaluating padding vector if data statistical info is applied to it.
            * if standardization/normalization transform is applied to padding, 'data_stats' should be provided
                'data_stats' format: {'shift': <torch.tenzor>, 'scale': <torch.tensor>} 
        """
        super().__init__(max_panel_len, data_stats)

    def __call__(self, predicted_panels, gt_panels=None):
        """
            Modified regression loss on panels: 
                * Allow panels to start from any origin vertex
        """
        # flatten input into list of panels
        if len(predicted_panels.shape) > 3:
            predicted_panels = predicted_panels.view(-1, predicted_panels.shape[-2], predicted_panels.shape[-1])
        if gt_panels is not None and len(gt_panels.shape) > 3:
            gt_panels = gt_panels.view(-1, gt_panels.shape[-2], gt_panels.shape[-1])

        # correct devices
        self.empty_panel_template = self.empty_panel_template.to(predicted_panels.device)
        self.pad_vector = self.pad_vector.to(predicted_panels.device)
        
        chosen_panels = []
        leading_edges = []
        gt_panel_num_edges = []
        # choose the closest version of original panel for each predicted panel
        with torch.no_grad():
            for el_id in range(predicted_panels.shape[0]):
                num_edges = panel_len_from_padded(gt_panels[el_id], empty_template=self.empty_panel_template)
                gt_panel_num_edges.append(num_edges)

                # all rotations of GT
                # TODO Faster version? -- I think I already did smth like this somewhere
                shifted_gt_panel = gt_panels[el_id]
                min_dist = ((predicted_panels[el_id] - shifted_gt_panel) ** 2).sum()
                chosen_panel = shifted_gt_panel
                leading_edge = 0
                for i in range(1, num_edges):
                    shifted_gt_panel = self._rotate_edges(shifted_gt_panel, num_edges)
                    dist = ((predicted_panels[el_id] - shifted_gt_panel) ** 2).sum()
                    if dist < min_dist:
                        min_dist = dist
                        chosen_panel = shifted_gt_panel
                        leading_edge = i

                # update choice
                chosen_panels.append(chosen_panel)
                leading_edges.append(leading_edge)

        chosen_panels = torch.stack(chosen_panels).to(predicted_panels.device)

        # batch mean of squared norms of per-panel final points:
        return nn.functional.mse_loss(predicted_panels, chosen_panels), leading_edges, gt_panel_num_edges
        
    def _rotate_edges(self, panel, num_edges):
        """
            Rotate the start of the loop to the next edge
        """
        panel = torch.cat((panel[1:num_edges], panel[0:1, :], panel[num_edges:]))

        return panel


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
            self, stitch_tags, free_edge_class, gt_stitches, gt_stitches_nums, 
            per_panel_leading_edges=None, gt_panel_num_edges=None, pattern_names=None):
        """
         Evaluate on the batch of stitch tags
        """
        # undo stats application if provided
        if self.data_stats is not None:
            device = stitch_tags.device
            stitch_tags = stitch_tags * self.data_stats['scale'].to(device) + self.data_stats['shift'].to(device)
        
        if per_panel_leading_edges is not None:  # re-arrange GT edge ids according to the match
            if gt_panel_num_edges is None:
                raise ValueError(
                    '{}::Error::need information on GT num of edges in every panel when *per_panel_leading_edges* is provided'.format(
                        self.__class__.__name__
                    )
                )
            max_num_panels = stitch_tags.shape[1]
            max_panel_len = stitch_tags.shape[-2]
            gt_stitches = PatternStitchLoss.gt_stitches_shift(
                gt_stitches, gt_stitches_nums, per_panel_leading_edges, gt_panel_num_edges, max_num_panels, max_panel_len)

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

    def __call__(self, predicted_outlines, gt_outlines, gt_panel_nums, pattern_names=None):
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
                # TODO switch to commond function
                predicted_bool_matrix = torch.isclose(
                    predicted_outlines[pattern_idx][panel_id], 
                    self.empty_panel_template, atol=0.07)  # this value is adjusted to have similar effect to what is used in core.py
                # empty panel detected -- stop further eval
                if torch.all(predicted_bool_matrix):
                    break

                # check is the num of edges matches
                predicted_num_edges = (~torch.all(predicted_bool_matrix, axis=1)).sum()  # only non-padded rows
            
                if predicted_num_edges < 3:
                    # 0, 1, 2 edges are not enough to form a panel -> assuming this is an empty panel
                    break
                # othervise, we have a real panel
                predicted_num_panels += 1

                gt_bool_matrix = torch.isclose(gt_outlines[pattern_idx][panel_id], self.empty_panel_template, atol=0.07)
                gt_num_edges = (~torch.all(gt_bool_matrix, axis=1)).sum()  # only non-padded rows

                panel_correct = (predicted_num_edges == gt_num_edges)
                correct_num_edges += panel_correct

                if pattern_names is not None and not panel_correct:  # pattern len predicted wrongly
                    print('NumbersInPanelsAccuracies::{}::panel {}:: {} edges instead of {}'.format(
                        pattern_names[pattern_idx], panel_id,
                        predicted_num_edges, gt_num_edges))
    
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
    
    def __call__(self, predicted_outlines, gt_outlines, per_panel_leading_edges=None):
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

            # unpad using correct gt info -- for simplicity of comparison
            num_edges = panel_len_from_padded(gt_panel, empty_template=self.empty_panel_template)
            if num_edges < 3:
                # 0, 1, 2 edges are not enough to form a panel -> assuming this is an empty panel
                break

            prediced_panel = prediced_panel[:num_edges, :]  
            gt_panel = gt_panel[:num_edges, :]

            if per_panel_leading_edges is not None:
                edge_id = per_panel_leading_edges[panel_idx]
                if edge_id:  # not zero -- shift needed
                    gt_panel = torch.cat((gt_panel[edge_id:], gt_panel[: edge_id]))

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


# ------- Model evaluation shortcut -------------
def eval_metrics(model, data_wrapper, section='test'):
    """Evalutes current model on the given dataset section"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    if hasattr(model, 'with_quality_eval'):
        model.with_quality_eval = True  # force quality evaluation for models that support it

    with torch.no_grad():
        loader = data_wrapper.get_loader(section)
        if isinstance(loader, dict):
            metrics_dict = {}
            for data_folder, loader in loader.items():
                metrics_dict[data_folder] = _eval_metrics_per_loader(model, loader, device)
            return metrics_dict
        else:
            return _eval_metrics_per_loader(model, loader, device)


def _eval_metrics_per_loader(model, loader, device):
    """
    Evaluate model on given loader. 
    
    Secondary function -- it assumes that context is set up: torch.no_grad(), model device & mode, etc."""

    current_metrics = dict.fromkeys(['full_loss'], 0)
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
