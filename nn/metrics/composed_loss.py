import torch
from torch._C import import_ir_module
import torch.nn as nn

from entmax import SparsemaxLoss  # https://github.com/deep-spin/entmax

import gap

# My modules
from data import Garment3DPatternFullDataset as PatternDataset
from pattern_converter import InvalidPatternDefError
from metrics.losses import *
from metrics.metrics import *


class ComposedLoss():
    """Base interface for compound loss objects"""

    def __init__(self, data_config, in_config={}):
        """
            Initialize loss components
            Accepts (in in_config):
            * Requested list of components
            * Additional configurations for losses (e.g. edge-origin agnostic evaluation)
            * data_stats -- for correct definition of losses
        """
        self.config = {  # defults
            'loss_components': [], 
            'quality_components': [],
        }
        self.config.update(in_config)  # override with requested settings

        self.with_quality_eval = True  # quality evaluation switch -- may allow to speed up the loss evaluation if False
        self.training = False  # training\evaluation state

        # Convenience properties
        self.l_components = self.config['loss_components']
        self.q_components = self.config['quality_components'] 

        if 'edge_pair_class' in self.l_components:
            self.bce_logits_loss = nn.BCEWithLogitsLoss()  # binary classification loss
        

    def __call__(self, preds, ground_truth, names=None, epoch=1000):
        """Evalute loss when predicting patterns.
            * Predictions are expected to follow the default GT structure, 
                but don't have to have all components -- as long as provided prediction is sufficient for
                evaluation of requested losses
            * default epoch is some large value to trigger stitch evaluation
            * Function returns True in third parameter at the moment of the loss stucture update
        """
        self.device = preds.device
        loss_dict = {}
        full_loss = 0.

        # match devices with prediction
        ground_truth = ground_truth.to(self.device)

        # ---- Losses ------
        main_losses, main_dict = self._main_losses(preds, ground_truth, None, epoch)
        full_loss += main_losses
        loss_dict.update(main_dict)

        # ---- Quality metrics  ----
        if self.with_quality_eval:
            with torch.no_grad():
                quality_breakdown = self._main_quality_metrics(preds, ground_truth, None, names)
                loss_dict.update(quality_breakdown)

        # final loss; breakdown for analysis; indication if the loss structure has changed on this evaluation
        return full_loss, loss_dict, False


    def eval(self):
        """ Loss to evaluation mode """
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def _main_losses(self, preds, ground_truth, gt_num_edges, epoch):
        """
            Main loss components. Evaluated in the same way regardless of the training stage
        """
        full_loss = 0.
        loss_dict = {}

        if 'edge_pair_class' in self.l_components:
            # flatten for correct computation
            pair_loss = self.bce_logits_loss(
                preds.view(-1), ground_truth.view(-1).type(torch.FloatTensor).to(self.device))
            loss_dict.update(edge_pair_class_loss=pair_loss)
            full_loss += pair_loss

        return full_loss, loss_dict

    def _main_quality_metrics(self, preds, ground_truth, gt_num_edges, names):
        """
            Evaluate quality components -- these are evaluated in the same way regardless of the training stage
        """
        loss_dict = {}
    
        if 'edge_pair_class' in self.q_components or 'edge_pair_stitch_recall' in self.q_components:
            edge_pair_class = torch.round(torch.sigmoid(preds))
            gt_mask = ground_truth.to(preds.device)

        if 'edge_pair_class' in self.q_components:
            acc = (edge_pair_class == gt_mask).sum().float() / gt_mask.numel()
            loss_dict.update(edge_pair_class_acc=acc)
        
        if 'edge_pair_stitch_recall' in self.q_components:
            prec, rec = self._prec_recall(edge_pair_class, gt_mask, target_label=1)
            loss_dict.update(stitch_precision=prec, stitch_recall=rec)

        return loss_dict

    def _prec_recall(self, preds, ground_truth, target_label):
        """ Evaluate precision/recall for given label in predictions """

        # correctly labeled as target label
        target_label_ids = (ground_truth == target_label).nonzero(as_tuple=True)
        correct_count = torch.count_nonzero(preds[target_label_ids] == target_label).float()

        # total number of labeled as target label
        pred_as_target_count = torch.count_nonzero(preds == target_label).float()

        # careful with division by zero
        precision = correct_count / pred_as_target_count if pred_as_target_count else 0
        recall = correct_count / len(target_label_ids[0]) if len(target_label_ids[0]) else 0

        return precision, recall


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
            'segm_loss_weight': 0.05,
            'stitch_tags_margin': 0.3,
            'epoch_with_stitches': 40, 
            'stitch_supervised_weight': 0.1,   # only used when explicit stitches are enabled
            'stitch_hardnet_version': False,

            'panel_origin_invariant_loss': True,
            'panel_order_inariant_loss': True,
            'order_by': 'placement',
            'epoch_with_order_matching': 0,

            'cluster_by': 'order_feature',  # 'panel_encodings', 'order_feature''   -- default is to use the same feature as for order matching
            'epoch_with_cluster_checks': 0,
            'gap_cluster_threshold': 0.,  # differentiating single\multi class cases 
            'diff_cluster_threshold': 0.,  # differentiating single\multi class cases 
            'cluster_gap_nrefs': 20,   # reducing will speed up training
            'cluster_with_singles': False, 

            'att_empty_weight': 1,  # for empty panels zero attention loss
            'att_distribution_saturation': 0.1,
            'epoch_with_att_saturation': 0
        }
        self.config.update(in_config)  # override with requested settings

        self.with_quality_eval = True  # quality evaluation switch -- may allow to speed up the loss evaluation if False
        self.training = False  # training\evaluation state
        self.debug_prints = False

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

        # store moving-around-clusters info
        self.cluster_resolution_mapping = {}

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
            self.bce_logits_loss = nn.BCEWithLogitsLoss()  # binary classification loss
        if 'att_distribution' in self.l_components:
            self.att_distribution = AttentionDistributionLoss(self.config['att_distribution_saturation'])
        if 'segmentation' in self.l_components:
            # Segmenation output is Sparsemax scores (not SoftMax), hence using the appropriate loss
            self.segmentation = SparsemaxLoss()

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
        self.epoch = epoch
        loss_dict = {}
        full_loss = 0.

        # match devices with prediction
        for key in ground_truth:
            ground_truth[key] = ground_truth[key].to(self.device)  

        # ------ GT pre-processing --------
        if self.config['panel_order_inariant_loss']:  # match panel order
            # NOTE: Not supported for 
            if 'segmentation' in self.l_components: 
                raise NotImplementedError('Order matching not supported for training with segmentation losses')
            gt_rotated, order_metrics = self._gt_order_match(preds, ground_truth) 
            loss_dict.update(order_metrics)
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
                quality_breakdown, corr_mask = self._main_quality_metrics(preds, gt_rotated, gt_num_edges, names)
                loss_dict.update(quality_breakdown)

                # stitches quality
                if epoch >= self.config['epoch_with_stitches']:
                    quality_breakdown = self._stitch_quality_metrics(
                        preds, gt_rotated, gt_num_edges, names, corr_mask)
                    loss_dict.update(quality_breakdown)

        loss_update_ind = (epoch == self.config['epoch_with_stitches'] and any((el in self.l_components for el in ['stitch', 'stitch_supervised', 'free_class']))
            or epoch == self.config['epoch_with_order_matching'] and self.config['panel_order_inariant_loss']
            or epoch == self.config['epoch_with_cluster_checks'] and self.config['cluster_by'] is not None) 

        # final loss; breakdown for analysis; indication if the loss structure has changed on this evaluation
        return full_loss, loss_dict, loss_update_ind

    def eval(self):
        """ Loss to evaluation mode """
        self.training = False

    def train(self, mode=True):
        self.training = mode

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

        if 'att_distribution' in self.l_components and self.epoch >= self.config['epoch_with_att_saturation']:
            att_loss = self.att_distribution(preds['att_weights'])
            full_loss += att_loss
            loss_dict.update(att_distribution_loss=att_loss)

        if 'segmentation' in self.l_components:

            # DEBUG
            # print(preds['att_weights'].shape, ground_truth['segmentation'].shape)
            pred_flat = preds['att_weights'].view(-1, preds['att_weights'].shape[-1])
            gt_flat = ground_truth['segmentation'].view(-1)
            # print(pred_flat.shape, gt_flat.shape)

            # DEBUG
            # with torch.no_grad():
            #     print(torch.sum(pred_flat, dim=1).shape)
            #     print(torch.mean(torch.sum(pred_flat, dim=1)))

            # print(pred_flat)
            # print('xxxx')
            # print(gt_flat, gt_flat.max(), gt_flat.min())
            # print('-----------')

            # NOTE!!! SparseMax produces exact zeros
            segm_loss = self.segmentation(pred_flat, gt_flat)

            # print(segm_loss)

            full_loss += self.config['segm_loss_weight'] * segm_loss
            loss_dict.update(segm_loss=segm_loss)

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
            free_edges_loss = self.bce_logits_loss(
                preds['free_edges_mask'], ground_truth['free_edges_mask'].type(torch.FloatTensor).to(self.device))
            loss_dict.update(free_edges_loss=free_edges_loss)
            full_loss += free_edges_loss

        return full_loss, loss_dict

    def _main_quality_metrics(self, preds, ground_truth, gt_num_edges, names):
        """
            Evaluate quality components -- these are evaluated in the same way regardless of the training stage
        """
        loss_dict = {}

        correct_mask = None
        if 'discrete' in self.q_components:
            num_panels_acc, num_edges_acc, correct_mask, num_edges_correct_acc = self.pattern_nums_quality(
                preds['outlines'], gt_num_edges, ground_truth['num_panels'], pattern_names=names)
            loss_dict.update(
                num_panels_accuracy=num_panels_acc, 
                num_edges_accuracy=num_edges_acc,
                corr_num_edges_accuracy=num_edges_correct_acc)

        if 'shape' in self.q_components:
            shape_l2, correct_shape_l2 = self.pattern_shape_quality(
                preds['outlines'], ground_truth['outlines'], gt_num_edges, correct_mask)
            loss_dict.update(
                panel_shape_l2=shape_l2, 
                corr_panel_shape_l2=correct_shape_l2, 
            )
        
        if 'rotation' in self.q_components:
            rotation_l2, correct_rotation_l2 = self.rotation_quality(
                preds['rotations'], ground_truth['rotations'], correct_mask)
            loss_dict.update(rotation_l2=rotation_l2, corr_rotation_l2=correct_rotation_l2)

        if 'translation' in self.q_components:
            translation_l2, correct_translation_l2 = self.translation_quality(
                preds['translations'], ground_truth['translations'], correct_mask)
            loss_dict.update(translation_l2=translation_l2, corr_translation_l2=correct_translation_l2)
    
        return loss_dict, correct_mask

    def _stitch_quality_metrics(self, preds, ground_truth, gt_num_edges, names, correct_mask):
        """
            Quality components related to stitches prediction. May be called separately from main components 
            arrording to the training stage
        """
        loss_dict = {}
        if 'stitch' in self.q_components:
            stitch_prec, stitch_recall, corr_prec, corr_rec = self.stitch_quality(
                preds['stitch_tags'], preds['free_edges_mask'], 
                ground_truth['stitches'].type(torch.IntTensor).to(self.device), 
                ground_truth['num_stitches'],
                pattern_names=names, 
                correct_mask=correct_mask)
            loss_dict.update(
                stitch_precision=stitch_prec, 
                stitch_recall=stitch_recall,
                corr_stitch_precision=corr_prec, 
                corr_stitch_recall=corr_rec)
        
        if 'free_class' in self.q_components:
            free_class = torch.round(torch.sigmoid(preds['free_edges_mask']))
            gt_mask = ground_truth['free_edges_mask'].to(preds['free_edges_mask'].device)
            acc = (free_class == gt_mask).sum().float() / gt_mask.numel()

            loss_dict.update(free_edge_acc=acc)

        return loss_dict

    # ------ Ground truth panel order match -----
    def _gt_order_match(self, preds, ground_truth):
        """
            Find the permutation of panel in GT that is best matched with the prediction (by geometry)
            and return the GT object with all properties updated according to this permutation 
        """
        with torch.no_grad():
            gt_updated = {}

            # Match the order
            if self.config['order_by'] == 'placement':
                if ('translations' not in preds 
                        or 'rotations' not in preds):
                    raise ValueError('ComposedPatternLoss::Error::Ordering by placement requested but placement is not predicted')

                pred_feature = torch.cat([preds['translations'], preds['rotations']], dim=-1)
                gt_feature = torch.cat([ground_truth['translations'], ground_truth['rotations']], dim=-1)

            elif self.config['order_by'] == 'translation':
                if 'translations' not in preds:
                    raise ValueError('ComposedPatternLoss::Error::Ordering by translation requested but translation is not predicted')
                
                pred_feature = preds['translations']
                gt_feature = ground_truth['translations']
                
            elif self.config['order_by'] == 'shape_translation':
                if 'translations' not in preds:
                    raise ValueError('ComposedPatternLoss::Error::Ordering by translation requested but translation is not predicted')

                pred_outlines_flat = preds['outlines'].contiguous().view(preds['outlines'].shape[0], preds['outlines'].shape[1], -1)
                gt_outlines_flat = ground_truth['outlines'].contiguous().view(preds['outlines'].shape[0], preds['outlines'].shape[1], -1)

                pred_feature = torch.cat([preds['translations'], pred_outlines_flat], dim=-1)
                gt_feature = torch.cat([ground_truth['translations'], gt_outlines_flat], dim=-1)
 
            elif self.config['order_by'] == 'stitches':
                if ('free_edges_mask' not in preds
                        or 'translations' not in preds 
                        or 'rotations' not in preds):
                    raise ValueError('ComposedPatternLoss::Error::Ordering by stitches requested but free edges mask or placement are not predicted')
                
                pred_feature = torch.cat([preds['translations'], preds['rotations']], dim=-1)
                gt_feature = torch.cat([ground_truth['translations'], ground_truth['rotations']], dim=-1)

                if self.epoch >= self.config['epoch_with_stitches']: 
                    # add free mask as feature
                    # flatten per-edge info into single vector
                    # push preficted mask score to 0-to-1 range
                    pred_mask = torch.round(torch.sigmoid(preds['free_edges_mask'])).view(
                        preds['free_edges_mask'].shape[0], preds['free_edges_mask'].shape[1], -1)

                    gt_mask = ground_truth['free_edges_mask'].view(
                        ground_truth['free_edges_mask'].shape[0], ground_truth['free_edges_mask'].shape[1], -1)

                    pred_feature = torch.cat([pred_feature, pred_mask], dim=-1)
                    gt_feature = torch.cat([gt_feature, gt_mask], dim=-1)

                else:
                    print('ComposedPatternLoss::Warning::skipped order match by stitch tags as stitch loss is not enabled')      
                
            else:
                raise NotImplemented('ComposedPatternLoss::Error::Ordering by requested feature <{}> is not implemented'.format(
                    self.config['order_by']
                ))

            # run the optimal permutation eval
            gt_permutation = self._panel_order_match(pred_feature, gt_feature)

            collision_swaps_stats = {}
            if self.epoch >= self.config['epoch_with_cluster_checks'] and self.config['cluster_by'] is not None:
                # remove panel types collision even it's not the best match with net output
                # enourages good separation of panel "classes" during training, but not needed at evaluation time

                if self.config['cluster_by'] == 'panel_encodings':
                    cluster_feature = preds['panel_encodings']
                elif self.config['cluster_by'] == 'translation':
                    cluster_feature = self._feature_permute(ground_truth['translations'], gt_permutation)  # !!
                else:  # order_feature -- default by the same feature as ordering
                    cluster_feature = self._feature_permute(gt_feature, gt_permutation)  # !!

                gt_permutation, collision_swaps_stats = self._att_cluster_analysis(
                    cluster_feature, gt_permutation, ground_truth['empty_panels_mask']
                )

            # Update gt info according to the permutation
            gt_updated['outlines'] = self._feature_permute(ground_truth['outlines'], gt_permutation)
            gt_updated['num_edges'] = self._feature_permute(ground_truth['num_edges'], gt_permutation)
            gt_updated['empty_panels_mask'] = self._feature_permute(ground_truth['empty_panels_mask'], gt_permutation)
            
            # Not supported
            # gt_updated['segmentation'] = self._feature_permute(ground_truth['segmentation'], gt_permutation)

            if 'rotation' in self.l_components:
                gt_updated['rotations'] = self._feature_permute(ground_truth['rotations'], gt_permutation)
            if 'translation' in self.l_components:
                gt_updated['translations'] = self._feature_permute(ground_truth['translations'], gt_permutation)
                
            if self.epoch >= self.config['epoch_with_stitches'] and (
                    'stitch' in self.l_components
                    or 'stitch_supervised' in self.l_components
                    or 'free_class' in self.l_components):  # if there is any stitch-related evaluation

                gt_updated['stitches'] = self._stitch_after_permute( 
                    ground_truth['stitches'], ground_truth['num_stitches'], 
                    gt_permutation, self.max_panel_len
                )
                gt_updated['free_edges_mask'] = self._feature_permute(ground_truth['free_edges_mask'], gt_permutation)
                
                if 'stitch_supervised' in self.l_components:
                    gt_updated['stitch_tags'] = self._feature_permute(ground_truth['stitch_tags'], gt_permutation)

            # keep the references to the rest of the gt data as is
            for key in ground_truth:
                if key not in gt_updated:
                    gt_updated[key] = ground_truth[key]

        return gt_updated, collision_swaps_stats

    def _panel_order_match(self, pred_features, gt_features):
        """
            Find the best-matching permutation of gt panels to the predicted panels (in panel order)
            based on the provided panel features
        """
        with torch.no_grad():
            batch_size = pred_features.shape[0]
            pat_len = gt_features.shape[1]

            if self.epoch < self.config['epoch_with_order_matching']:
                # assign ordering randomly -- all the panel in the NN output have some non-zero signals at some point
                per_pattern_permutation = torch.stack(
                    [torch.randperm(pat_len, dtype=torch.long, device=pred_features.device) for _ in range(batch_size)]
                )
                return per_pattern_permutation

            # evaluate best order match
            # distances between panels (vectorized)
            total_dist_matrix = torch.cdist(
                pred_features.view(batch_size, pat_len, -1),   # flatten feature
                gt_features.view(batch_size, pat_len, -1))
            total_dist_flat_view = total_dist_matrix.view(batch_size, -1)

            # Assingment (vectorized in batch dimention)
            per_pattern_permutation = torch.full((batch_size, pat_len), fill_value=-1, dtype=torch.long, device=pred_features.device)
            for _ in range(pat_len):  # this many pair to arrange
                to_match_ids = total_dist_flat_view.argmin(dim=1)  # current global min is also a best match for the pair it's calculated for!
                
                rows = to_match_ids // total_dist_matrix.shape[1]
                cols = to_match_ids % total_dist_matrix.shape[1]

                for i in range(batch_size):  # only the easy operation is left unvectorized
                    per_pattern_permutation[i, rows[i]] = cols[i]
                    # exlude distances with matches
                    total_dist_matrix[i, rows[i], :] = float('inf')
                    total_dist_matrix[i, :, cols[i]] = float('inf')

            if torch.isfinite(total_dist_matrix).any():
                raise ValueError('ComposedPatternLoss::Error::Failed to match panel order')
        
        return per_pattern_permutation

    @staticmethod
    def _feature_permute(pattern_features, permutation):
        """
            Permute all given features (in the batch) according to given panel order permutation
        """
        with torch.no_grad():
            extended_permutation = permutation
            # match indexing with feature size
            if len(permutation.shape) < len(pattern_features.shape):
                for _ in range(len(pattern_features.shape) - len(permutation.shape)):
                    extended_permutation = extended_permutation.unsqueeze(-1)
                # expand just creates a new view without extra copies
                extended_permutation = extended_permutation.expand(pattern_features.shape)

            # collect features with correct permutation in pattern dimention
            indexed_features = torch.gather(pattern_features, dim=1, index=extended_permutation)
        
        return indexed_features

    @staticmethod
    def _stitch_after_permute(stitches, stitches_num, permutation, max_panel_len):
        """
            Update edges ids in stitch info after panel order permutation
        """
        with torch.no_grad():  # GT updates don't require gradient compute
            # add pattern dimention
            for pattern_id in range(len(stitches)):
                
                # inverse permutation for this pattern for faster access
                new_panel_ids_list = [-1] * permutation.shape[1]
                for i in range(permutation.shape[1]):
                    new_panel_ids_list[permutation[pattern_id][i]] = i

                # re-assign GT edge ids according to shift
                for side in (0, 1):
                    for i in range(stitches_num[pattern_id]):
                        edge_id = stitches[pattern_id][side][i]
                        panel_id = edge_id // max_panel_len
                        in_panel_edge_id = edge_id - (panel_id * max_panel_len)

                        # where is this panel placed
                        new_panel_id = new_panel_ids_list[panel_id]

                        # update with pattern-level edge id
                        stitches[pattern_id][side][i] = new_panel_id * max_panel_len + in_panel_edge_id
                
        return stitches

    # ------ Cluster analysis -------
    def _att_cluster_analysis(self, features, permutation, empty_panel_mask):
        """ (try) to find and resolve cases when multiple panel clusters 
            were assigned to the same panel id (hence attention slot)
            in the given batch
            Clusters are evaluated in unsupervised manner based on fiven feature
        """
        # Apply permutation (since we need to check the quality of permutation, cap =))
        empty_mask = self._feature_permute(empty_panel_mask, permutation)

        # references to non-empty elements only
        non_empty = ~empty_mask
        non_empty_ids_per_slot = []
        for panel_id in range(empty_mask.shape[-1]):
            non_empty_ids_per_slot.append(torch.nonzero(non_empty[:, panel_id], as_tuple=False).squeeze(-1))

        # evaluate clustering
        single_class, multiple_classes, empty_att_slots = self._eval_clusters(
            features, non_empty_ids_per_slot, max_k=self.max_pattern_size)

        avg_classes = float(sum([multiple_classes[el][0] for el in multiple_classes]) + len(single_class)) \
                      / (len(multiple_classes) + len(single_class))

        # update permulation for multi-class cases
        num_swaps = 0
        if len(multiple_classes):
            new_permutation, num_swaps = self._distribute_clusters(
                single_class, multiple_classes, empty_att_slots, non_empty_ids_per_slot, permutation)
        
        # updated permutation (if in training mode!!) & logging info
        return new_permutation if self.training and len(multiple_classes) else permutation, {
            'order_collision_swaps': num_swaps, 
            # Average "baddness" of the multi cluster cases, including zeros for single classes. 
            'multi-class-diffs': sum([multiple_classes[el][1] for el in multiple_classes]) / (len(multiple_classes) + len(single_class)) if len(multiple_classes) else 0, 
            'multiple_classes_on_cluster': float(len(multiple_classes)) / empty_mask.shape[-1],
            'avg_clusters': avg_classes
        }

    def _eval_clusters(self, features, non_empty_ids_per_slot, max_k=2):
        """
            Evaluate clustering property of given feature distribution for each panel id
        """
        empty_att_slots = []
        single_class = []  # TODO Single classes as dict too
        multiple_classes = {}
        for panel_id in range(features.shape[1]):
            non_empty_ids = non_empty_ids_per_slot[panel_id]
            if len(non_empty_ids) == 0:  
                # all panels at this place are empty
                empty_att_slots.append(panel_id)
                continue

            slot_features = features[non_empty_ids, panel_id, :]
            if len(non_empty_ids) == 1:  # one example -- one cluster, Captain!
                single_class.append((panel_id, slot_features[0], torch.cat((slot_features, slot_features))))
                continue
            
            # Differentiate single cluster from multi-cluster cases based on gap statistic         
            k_optimal, diff, labels, cluster_centers = gap.optimal_clusters(
                slot_features,
                max_k=max_k, 
                sencitivity_threshold=self.config['diff_cluster_threshold'], 
                logs=self.debug_prints
            )

            if k_optimal == 1:  # the last comes from gap stats formula
                single_class.append((panel_id, cluster_centers[0], self._bbox(slot_features)))

                if self.config['cluster_with_singles'] or panel_id in self.cluster_resolution_mapping:  
                    # allow to map to this panel_id is singles are in use
                    # AND make sure that this mapping is up-to-date when slot starts to be used
                    self.cluster_resolution_mapping[panel_id] = single_class[-1][2]

            else:
                # TODO don't caculate this cdist twice!
                # TODO Just max dist between features??
                diff = torch.cdist(cluster_centers, cluster_centers).max()
                bboxes = []
                for label_id in range(k_optimal):
                    if len(slot_features[labels == label_id]) == 0:  # happens, but rarely
                        bboxes.append(None)
                    else:
                        bboxes.append(
                            self._bbox(slot_features[labels == label_id])
                        )
                multiple_classes[panel_id] = (k_optimal, diff, labels, cluster_centers, bboxes)
            
            if self.debug_prints:
                print(panel_id, ' -- ', k_optimal, ' ', diff)

        if self.debug_prints:
            print('Single class: {}; Multi-class: {}; Empty: {};'.format(
                [el[0] for el in single_class], multiple_classes.keys(), empty_att_slots))

        return single_class, multiple_classes, empty_att_slots

    def _distribute_clusters(self, single_class, multiple_classes, empty_att_slots, non_empty_ids_per_slot, permutation):
        """
            Re-Distribute clusters of features in the dicovered multi-clustering cases
        """
        # Check if assignment could be reused 
        assigned = set()

        # Logging
        single_slots = [elem[0] for elem in single_class]
 
        # check if multi-class could be assigned to the same slot \ used single slot as earlier
        memory_slots = []
        memory_bboxes = []
        for k in self.cluster_resolution_mapping:
            if k in empty_att_slots:
                memory_slots.append(k)
                memory_bboxes.append(self.cluster_resolution_mapping[k])
        if len(memory_slots):
            for current_slot in multiple_classes:
                if current_slot not in assigned:
                    k, curr_quality, labels, m_cluster_centers, m_bboxes = multiple_classes[current_slot]

                    # comparison of current slot clusters to single-cluster slots
                    bb_comparion = torch.zeros((k, len(memory_slots)), device=permutation.device)
                    for label_id, m_box in enumerate(m_bboxes):
                        if m_box is None:  # skip
                            continue
                        for slot_idx in range(len(memory_slots)):
                            bbox_iou = self._bbox_iou(m_box, memory_bboxes[slot_idx])  # with single mox
                            bb_comparion[label_id, slot_idx] = bbox_iou

                    if bb_comparion.max() > 0.8:  # TODO parameter??
                        flat_idx = bb_comparion.argmax()
                        label_id = flat_idx // bb_comparion.shape[1]
                        single_slot_list_id = flat_idx - label_id * bb_comparion.shape[1]
                        new_slot = memory_slots[single_slot_list_id]

                        # Update
                        try:
                            permutation = self._swap_slots(
                                permutation, labels, non_empty_ids_per_slot, label_id, current_slot, new_slot)
                        except ValueError as e:
                            if self.debug_prints:
                                print(e)
                            continue

                        if self.debug_prints:
                            tag = 'Using single' if new_slot in single_slots else 'Re-using'
                            print(
                                f'{tag} {current_slot}->{new_slot} with iou {bb_comparion[label_id, single_slot_list_id]:.4f}'
                                f' by {m_bboxes[label_id]} with '
                                f'original {memory_bboxes[single_slot_list_id]}')
                        
                        # Logging Info
                        if new_slot in empty_att_slots:
                            empty_att_slots.remove(new_slot)
                        assigned.add(current_slot)

        # All the others are put to the new slots
        for current_slot in multiple_classes:
            if current_slot not in assigned and len(empty_att_slots):  # have options
                k, curr_quality, labels, m_cluster_centers, m_bboxes = multiple_classes[current_slot]

                new_slot = empty_att_slots.pop(0)  # use first available empty slot
                assigned.add(current_slot)
                if self.debug_prints:
                    print(f'Using Empty {current_slot}->{new_slot}')  # Trying string interpolation

                # Choose elements to move
                if k > 2:  # the cluster that is further away from others
                    dists = torch.cdist(m_cluster_centers, m_cluster_centers).sum(dim=-1)
                    label_id = dists.argmax()
                else:  # or the one used the least -- in case of 2 classes
                    histogram = torch.histc(labels, bins=k, max=(k - 1))
                    label_id = histogram.argmin()

                # record for re-use. The m_bbox should not be None here..
                self.cluster_resolution_mapping[new_slot] = m_bboxes[label_id]

                # Update
                permutation = self._swap_slots(permutation, labels, non_empty_ids_per_slot, label_id, current_slot, new_slot)
        
        if self.debug_prints:
            print('After upds: Single class: {}; Multi-class: {}; Assigned: {}, Empty: {};'.format(
                [el[0] for el in single_class], multiple_classes.keys(), assigned, empty_att_slots))
        
        return (permutation, len(assigned))

    @staticmethod
    def _swap_slots(permutation, labels, non_empty_ids_per_slot, label_id, current_slot, new_slot):
        # move non-empy panels from current_slot to empty_slot in permutation
        indices = (labels == label_id).nonzero(as_tuple=False).squeeze(-1)
        indices = non_empty_ids_per_slot[current_slot][indices]  # convert to ids in batch 

        target_overlap = [idx for idx in indices if idx in non_empty_ids_per_slot[new_slot]]
        if len(target_overlap) > 0:
            raise ValueError(f'Tried to swap {current_slot}->{new_slot} with non-empty elements in {new_slot}: {target_overlap}. ')

        permutation[indices, current_slot], permutation[indices, new_slot] = permutation[indices, new_slot], permutation[indices, current_slot]

        return permutation

    @staticmethod
    def _bbox(features):
        """Evalyate bbox for given features"""
        return torch.stack((features.min(dim=0)[0], features.max(dim=0)[0]))

    @staticmethod
    def _bbox_iou(bbox_1, bbox_2, tol=0.001):
        """ 
            Compute IOU for 2 n-dimentional BBoxes (assuming they are aligned with coordinate frame)
            bbox[0] -- min, bbox[1] -- max
        """
        min_max_tols = torch.tensor([-tol, tol], device=bbox_1.device).unsqueeze(1).expand(-1, bbox_1.shape[-1])

        bbox_intersect = [torch.max(bbox_1[0], bbox_2[0]) - tol, torch.min(bbox_1[0], bbox_2[0]) + tol]
        intersect_volume = ComposedPatternLoss._bbox_volume(bbox_intersect)

        union_volume = (ComposedPatternLoss._bbox_volume(bbox_1 + min_max_tols)
                        + ComposedPatternLoss._bbox_volume(bbox_2 + min_max_tols)
                        - intersect_volume)
        
        return intersect_volume / union_volume

    @staticmethod
    def _bbox_volume(bboxs):
        diffs = bboxs[1] - bboxs[0]
        if (diffs < 0).any():  # inverted bbox
            return 0.

        return diffs.prod()

    # ------ Ground truth panel edge loop origin shift  ---------
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

