import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax
import wandb as wb

# my modules
from metrics.composed_loss import ComposedLoss, ComposedPatternLoss
import net_blocks as blocks


# ------ Basic Interface --------
# TODO no need for this class except for the model name!!
class BaseModule(nn.Module):
    """Base interface for my neural nets"""
    def __init__(self):
        super().__init__()
        self.config = {
            'loss': 'MSELoss',
            'model': self.__class__.__name__
        }
        self.regression_loss = nn.MSELoss()
    
    def loss(self, preds, ground_truth, **kwargs):
        """Default loss for my neural networks. Takes pne batch of data. 
            Children can use additional arguments as needed
        """
        ground_truth = ground_truth.to(preds.device)  # make sure device is correct
        loss = self.regression_loss(preds, ground_truth)
        return loss, {'regression loss': loss}, False  # second term is for compound losses, third -- to indicate dynamic update of loss structure


    def train(self, mode=True):
        super().train(mode)
        if isinstance(self.loss, object):
            self.loss.train(mode)
    
    def eval(self):
        super().eval()
        if isinstance(self.loss, object):
            self.loss.eval()


# -------- Nets architectures -----------
# -------- AEs--------------
class GarmentPanelsAE(BaseModule):
    """
        Model to test encoding & decoding of garment 2D panels (sewing patterns components)
        * Follows similar structure of GarmentFullPattern3D
        
    """
    def __init__(self, data_config, config={}, in_loss_config={}):
        super().__init__()

        # data props
        self.panel_elem_len = data_config['element_size']
        self.max_panel_len = data_config['max_panel_len']
        self.max_pattern_size = data_config['max_pattern_len']

        # ---- Net configuration ----
        self.config.update({
            'panel_encoding_size': 20, 
            'panel_n_layers': 3, 
            'pattern_encoding_size': 40, 
            'pattern_n_layers': 3,
            'dropout': 0,
            'lstm_init': 'kaiming_normal_', 
            'decoder': 'LSTMDecoderModule'
        })
        # update with input settings
        self.config.update(config) 

        # --- Losses ---
        self.config['loss'] = {
            'loss_components': ['shape', 'loop'],
            'quality_components': ['shape', 'discrete'],
            'panel_origin_invariant_loss': True,
            'loop_loss_weight': 0.1
        }
        self.config['loss'].update(in_loss_config)  # apply input settings 

        # create loss!
        self.loss = ComposedPatternLoss(data_config, self.config['loss'])
        self.config['loss'] = self.loss.config  # sync

        # ------ Modules ----
        decoder_module = getattr(blocks, self.config['decoder'])

        # --- panel-level ---- 
        self.panel_encoder = blocks.LSTMEncoderModule(
            self.panel_elem_len, 
            self.config['panel_encoding_size'], 
            self.config['panel_n_layers'], 
            dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )
        self.panel_decoder = decoder_module(
            self.config['panel_encoding_size'], 
            self.config['panel_encoding_size'], 
            self.panel_elem_len, 
            self.config['panel_n_layers'], 
            out_len=self.max_panel_len,
            dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )   

    def forward_encode(self, patterns_batch):
        """
            Predict garment encodings for input batch
            In this case it's just a concatenation of the panel encodings
        """
        self.batch_size = patterns_batch.size(0)
        self.max_pattern_size = patterns_batch.size(1)

        # flatten -- view simply as a list of panels to apply encoding per panel
        all_panels = patterns_batch.contiguous().view(-1, patterns_batch.shape[-2], patterns_batch.shape[-1])
        panel_encodings = self.panel_encoder(all_panels)

        # group by patterns -- garments encoding as concatenated panel encodings
        panel_encodings = panel_encodings.contiguous().view(self.batch_size, -1) 

        return panel_encodings 

    def forward_pattern_decode(self, garment_encodings):
        """
            Unfold provided garment encodings into per-panel encodings
            Useful for obtaining the latent space for Panels
            NOTE In this class, garment is represented as concatenation of panel encodings already, 
            so it just needs re-shaping it
        """
        return garment_encodings.contiguous().view(-1, self.config['panel_encoding_size'])

    def forward(self, patterns_batch, **kwargs):
        self.device = patterns_batch.device
        self.batch_size = patterns_batch.size(0)

        # --- Encode ---
        all_panels = patterns_batch.contiguous().view(-1, patterns_batch.shape[-2], patterns_batch.shape[-1])
        flat_panel_encodings = self.panel_encoder(all_panels)

        # --- Decode ---
        flat_panels_dec = self.panel_decoder(flat_panel_encodings, self.max_panel_len)

        # back to patterns and panels structure
        prediction = flat_panels_dec.contiguous().view(self.batch_size, self.max_pattern_size, self.max_panel_len, -1)
        
        return {'outlines': prediction}


class GarmentPatternAE(GarmentPanelsAE):
    """
        Hierarchical Sewing pattern AE -- defines garment-level pattern 
        * loss evaluation is the same for both AE models
    """
    def __init__(self, data_config, config={}, in_loss_config={}):
        super().__init__(data_config, config, in_loss_config)
        # loss objects & config already defined

        # adding patten-level encoders\decoders
        # ----- patten level ------
        self.pattern_encoder = blocks.LSTMEncoderModule(
            self.config['panel_encoding_size'], 
            self.config['pattern_encoding_size'], 
            self.config['pattern_n_layers'], 
            dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )
        decoder_module = getattr(blocks, self.config['decoder'])
        self.pattern_decoder = decoder_module(
            self.config['pattern_encoding_size'], 
            self.config['pattern_encoding_size'], 
            self.config['panel_encoding_size'], 
            self.config['pattern_n_layers'], 
            out_len=self.max_pattern_size,
            dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )

    def forward_encode(self, patterns_batch):
        """
            Predict garment encodings for input batch
        """
        self.batch_size = patterns_batch.size(0)
        self.max_pattern_size = patterns_batch.size(1)

        # flatten -- view simply as a list of panels to apply encoding per panel
        all_panels = patterns_batch.contiguous().view(-1, patterns_batch.shape[-2], patterns_batch.shape[-1])
        panel_encodings = self.panel_encoder(all_panels)

        panel_encodings = panel_encodings.contiguous().view(self.batch_size, self.max_pattern_size, -1)  # group by patterns
        pattern_encoding = self.pattern_encoder(panel_encodings)   # YAAAAY Pattern hidden representation!!

        return pattern_encoding 

    def forward_pattern_decode(self, garment_encodings):
        """
            Unfold provided garment encodings into per-panel encodings
            Useful for obtaining the latent space for Panels
        """
        panel_encodings = self.pattern_decoder(garment_encodings, self.max_pattern_size)
        flat_panel_encodings = panel_encodings.contiguous().view(-1, panel_encodings.shape[-1])

        return flat_panel_encodings

    def forward(self, patterns_batch, **kwargs):
        self.device = patterns_batch.device
        self.batch_size = patterns_batch.size(0)

        # --- Encode ---
        pattern_encoding = self.forward_encode(patterns_batch)

        # --- Decode ---
        flat_panel_encodings_dec = self.forward_pattern_decode(pattern_encoding)
        flat_panels_dec = self.panel_decoder(flat_panel_encodings_dec, self.max_panel_len)

        # back to patterns and panels
        prediction = flat_panels_dec.contiguous().view(self.batch_size, self.max_pattern_size, self.max_panel_len, -1)
        
        return {'outlines': prediction}


# ------------ Pattern predictions ----------
class GarmentFullPattern3D(BaseModule):
    """
        Predicting 2D pattern inluding panel placement and stitches information from 3D garment geometry 
        Constists of 
            * (interchangeable) feature extractor 
            * pattern decoder from GarmentPatternAE
            * MLP modules to predict panel 3D placement & stitches
    """
    def __init__(self, data_config, config={}, in_loss_config={}):
        super().__init__()

        # output props
        self.panel_elem_len = data_config['element_size']
        self.max_panel_len = data_config['max_panel_len']
        self.max_pattern_size = data_config['max_pattern_len']
        self.rotation_size = data_config['rotation_size']
        self.translation_size = data_config['translation_size']

        # ---- Net configuration ----
        self.config.update({
            'panel_encoding_size': 70, 
            'panel_hidden_size': 70,
            'panel_n_layers': 4, 
            'pattern_encoding_size': 130, 
            'pattern_hidden_size': 130, 
            'pattern_n_layers': 3, 
            'dropout': 0,
            'lstm_init': 'kaiming_normal_', 
            'feature_extractor': 'EdgeConvFeatures',
            'panel_decoder': 'LSTMDecoderModule', 
            'pattern_decoder': 'LSTMDecoderModule', 
            'stitch_tag_dim': 3
        })
        # update with input settings
        self.config.update(config) 

        # ---- losses configuration ----
        self.config['loss'] = {
            'loss_components': ['shape', 'loop', 'rotation', 'translation'],  # , 'stitch', 'free_class'],
            'quality_components': ['shape', 'discrete', 'rotation', 'translation'],  #, 'stitch', 'free_class'],
            'panel_origin_invariant_loss': True,
            'loop_loss_weight': 1.,
            'stitch_tags_margin': 0.3,
            'epoch_with_stitches': 40, 
            'stitch_supervised_weight': 0.1,   # only used when explicit stitch loss is used
            'stitch_hardnet_version': False,
            'panel_origin_invariant_loss': True
        }
        self.config['loss'].update(in_loss_config)
        # loss object
        self.loss = ComposedPatternLoss(data_config, self.config['loss'])
        self.config['loss'] = self.loss.config  # sync just in case

        # ---- Feature extractor definition -------
        feature_extractor_module = getattr(blocks, self.config['feature_extractor'])
        self.feature_extractor = feature_extractor_module(self.config['pattern_encoding_size'], self.config)
        if hasattr(self.feature_extractor, 'config'):
            self.config.update(self.feature_extractor.config)   # save extractor's additional configuration

        # ----- Decode into pattern definition -------
        panel_decoder_module = getattr(blocks, self.config['panel_decoder'])
        self.panel_decoder = panel_decoder_module(
            encoding_size=self.config['panel_encoding_size'], 
            hidden_size=self.config['panel_hidden_size'], 
            out_elem_size=self.panel_elem_len + self.config['stitch_tag_dim'] + 1,  # last element is free tag indicator 
            n_layers=self.config['panel_n_layers'], 
            out_len = self.max_panel_len,
            dropout=self.config['dropout'], 
            custom_init=self.config['lstm_init']
        )
        pattern_decoder_module = getattr(blocks, self.config['pattern_decoder'])
        self.pattern_decoder = pattern_decoder_module(
            encoding_size=self.config['pattern_encoding_size'], 
            hidden_size=self.config['pattern_hidden_size'], 
            out_elem_size=self.config['panel_encoding_size'], 
            n_layers=self.config['pattern_n_layers'], 
            out_len=self.max_pattern_size,
            dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )

        # decoding the panel placement
        self.placement_decoder = nn.Linear(
            self.config['panel_encoding_size'], 
            self.rotation_size + self.translation_size)

    def forward_encode(self, positions_batch):
        """
            Predict garment encodings for input point coulds batch
        """
        return self.feature_extractor(positions_batch)[0]  # YAAAAY Pattern hidden representation!!

    def forward_pattern_decode(self, garment_encodings):
        """
            Unfold provided garment encodings into per-panel encodings
            Useful for obtaining the latent space for Panels
        """
        panel_encodings = self.pattern_decoder(garment_encodings, self.max_pattern_size)
        flat_panel_encodings = panel_encodings.contiguous().view(-1, panel_encodings.shape[-1])

        return flat_panel_encodings

    def forward_panel_decode(self, flat_panel_encodings, batch_size):
        """ Panel encodings to outlines & stitch info """
        flat_panels = self.panel_decoder(flat_panel_encodings, self.max_panel_len)
        
        # Placement
        flat_placement = self.placement_decoder(flat_panel_encodings)
        flat_rotations = flat_placement[:, :self.rotation_size]
        flat_translations = flat_placement[:, self.rotation_size:]

        # reshape back to per-pattern predictions
        panel_predictions = flat_panels.contiguous().view(batch_size, self.max_pattern_size, self.max_panel_len, -1)
        stitch_tags = panel_predictions[:, :, :, self.panel_elem_len:-1]
        free_edge_class = panel_predictions[:, :, :, -1]
        outlines = panel_predictions[:, :, :, :self.panel_elem_len]

        rotations = flat_rotations.contiguous().view(batch_size, self.max_pattern_size, -1)
        translations = flat_translations.contiguous().view(batch_size, self.max_pattern_size, -1)

        return {
            'outlines': outlines, 
            'rotations': rotations, 'translations': translations, 
            'stitch_tags': stitch_tags, 'free_edges_mask': free_edge_class}

    def forward_decode(self, garment_encodings):
        """
            Unfold provided garment encodings into the sewing pattens
        """
        flat_panel_encodings = self.forward_pattern_decode(garment_encodings)

        return self.forward_panel_decode(flat_panel_encodings, garment_encodings.size(0))

    def forward(self, positions_batch, **kwargs):
        # Extract info from geometry 
        pattern_encodings = self.forward_encode(positions_batch)

        # Decode 
        return self.forward_decode(pattern_encodings)


class GarmentFullPattern3DDisentangle(GarmentFullPattern3D):
    """
        Predicting 2D pattern inluding panel placement and stitches information from 3D garment geometry 
        Constists of 
            * (interchangeable) feature extractor 
            * pattern decoder from GarmentPatternAE
            * MLP modules to predict panel 3D placement & stitches
        * Attempt to disentangle panel latent space
    """
    def __init__(self, data_config, config={}, in_loss_config={}):
        super().__init__(data_config, config, in_loss_config)

        # ----- Update panel decoders to promote space disentanglement -------
        self.placement_size = self.rotation_size + self.translation_size
        self.panel_shape_enc_size = int((self.config['panel_encoding_size'] - self.placement_size) / 2)
        self.stitch_enc_size = self.config['panel_encoding_size'] - self.placement_size - self.panel_shape_enc_size

        panel_decoder_module = getattr(blocks, self.config['panel_decoder'])
        self.panel_decoder = panel_decoder_module(
            self.panel_shape_enc_size, int(self.config['panel_encoding_size'] / 2), 
            self.panel_elem_len,  # last element is free tag indicator 
            self.config['panel_n_layers'], 
            out_len=self.max_panel_len,
            dropout=self.config['dropout'], 
            custom_init=self.config['lstm_init']
        )

        self.stitch_decoder = panel_decoder_module(
            self.stitch_enc_size, int(self.config['panel_encoding_size'] / 2), 
            self.config['stitch_tag_dim'] + 1,  # last element is free tag indicator 
            self.config['panel_n_layers'], 
            dropout=self.config['dropout'], 
            custom_init=self.config['lstm_init']
        )

        # decoding the panel placement
        self.placement_decoder = nn.Linear(
            self.placement_size, 
            self.placement_size)

    def forward_pattern_decode(self, garment_encodings):
        """
            Unfold provided garment encodings into per-panel encodings
            Useful for obtaining the latent space for Panels
        """
        panel_encodings = self.pattern_decoder(garment_encodings, self.max_pattern_size)
        flat_panel_encodings = panel_encodings.contiguous().view(-1, panel_encodings.shape[-1])

        # with removed placement prediction
        return flat_panel_encodings[:, (self.placement_size + self.stitch_enc_size):]

    def forward_decode(self, garment_encodings):
        """
            Unfold provided garment encodings into the sewing pattens
        """
        self.device = garment_encodings.device
        batch_size = garment_encodings.size(0)

        panel_encodings = self.pattern_decoder(garment_encodings, self.max_pattern_size)
        flat_panel_encodings = panel_encodings.contiguous().view(-1, panel_encodings.shape[-1])

        flat_placement_encodings = flat_panel_encodings[:, :self.placement_size]
        flat_stitch_encodings = flat_panel_encodings[:, self.placement_size: (self.placement_size + self.stitch_enc_size)]
        flat_panel_encodings = flat_panel_encodings[:, (self.placement_size + self.stitch_enc_size):]

        # Panel outlines & stitch info
        flat_panels = self.panel_decoder(flat_panel_encodings, self.max_panel_len)
        flat_stitches = self.stitch_decoder(flat_stitch_encodings, self.max_panel_len)
        
        # Placement
        flat_placement = self.placement_decoder(flat_placement_encodings)
        flat_rotations = flat_placement[:, :self.rotation_size]
        flat_translations = flat_placement[:, self.rotation_size:]

        # reshape back to per-pattern predictions
        panel_predictions = flat_panels.contiguous().view(batch_size, self.max_pattern_size, self.max_panel_len, -1)
        stitch_predictions = flat_stitches.contiguous().view(batch_size, self.max_pattern_size, self.max_panel_len, -1)
        stitch_tags = stitch_predictions[:, :, :, 1:]
        free_edge_class = stitch_predictions[:, :, :, 0]

        rotations = flat_rotations.contiguous().view(batch_size, self.max_pattern_size, -1)
        translations = flat_translations.contiguous().view(batch_size, self.max_pattern_size, -1)

        return {
            'outlines': panel_predictions, 
            'rotations': rotations, 'translations': translations, 
            'stitch_tags': stitch_tags, 'free_edges_mask': free_edge_class}


class GarmentAttentivePattern3D(GarmentFullPattern3D):
    """
        Patterns from 3D data with point-level attention.
        Forward functions are subdivided for convenience of latent space inspection
    """
    def __init__(self, data_config, config={}, in_loss_config={}):
        super().__init__(data_config, config, in_loss_config)

        # set to true to get attention weights with prediction -- for visualization
        # Keep false in all unnecessary cases to save memory!
        self.save_att_weights = False 

        # ---- per-point attention module ---- 
        if 'attention_token_size' not in self.config:
            self.config['attention_token_size'] = 100  # default

        # taking in per-point features and previous encodings, outputting point weight
        pattern_decoder_module = getattr(blocks, self.config['pattern_decoder'])
        self.pattern_attention_decode = pattern_decoder_module(
            self.config['pattern_encoding_size'], self.config['pattern_encoding_size'], self.config['attention_token_size'], self.config['pattern_n_layers'], 
            out_len = self.max_pattern_size,
            dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )

        attention_input_size = self.config['attention_token_size'] + self.feature_extractor.config['EConv_feature']
        self.point_attention_mlp = nn.Sequential(
            blocks.MLP([attention_input_size, attention_input_size, attention_input_size, 1]),
            nn.Sigmoid()
        )

        # additional panel encoding post-procedding
        self.panel_dec_lin = nn.Linear(
            self.feature_extractor.config['EConv_feature'], self.feature_extractor.config['panel_encoding_size'])

        # pattern decoder is not needed any more
        del self.pattern_decoder

    def forward_panel_enc_from_3d(self, positions_batch):
        """
            Get per-panel encodings from 3D data directly
            
        """
        # ------ Point cloud features -------
        batch_size = positions_batch.shape[0]
        # per-point and total encodings
        init_pattern_encodings, point_features_flat, batch = self.feature_extractor(positions_batch)
        num_points = point_features_flat.shape[0] // batch_size

        # ---- attention indicators for each (future) panel ----- 
        attention_tokens = self.pattern_attention_decode(init_pattern_encodings, self.max_pattern_size)
        attention_tokens_flat = attention_tokens.view([-1, attention_tokens.shape[-1]])

        # ----- Getting per-panel features after attention application ------
        all_panel_features = []
        all_att_weights = []
        for panel_id in range(attention_tokens.shape[1]):
            # per-panel token
            panel_att_tokens = attention_tokens[:, panel_id, :]
            panel_att_tokens_flat = panel_att_tokens.view([-1, panel_att_tokens.shape[-1]])
            # propagate per-point
            panel_att_tokens_flat = panel_att_tokens_flat.unsqueeze(1).repeat(1, num_points, 1).view([-1, panel_att_tokens.shape[-1]])

            # concat with features and get weights
            att_weights = self.point_attention_mlp(torch.cat([panel_att_tokens_flat, point_features_flat], dim=-1))

            if self.save_att_weights:
                all_att_weights.append(att_weights.view(batch_size, -1))

            # weight and pool to get panel encoding
            weighted_features = att_weights * point_features_flat

            # same pool as in intial extractor
            panel_feature = self.feature_extractor.global_pool(weighted_features, batch, batch_size) 
            panel_feature = self.panel_dec_lin(panel_feature)  # reshape as needed
            panel_feature = panel_feature.view(batch_size, -1, panel_feature.shape[-1])

            all_panel_features.append(panel_feature)

        panel_encodings = torch.cat(all_panel_features, dim=1)  # concat in pattern dimention

        if len(all_att_weights) > 0:
            all_att_weights = torch.stack(all_att_weights, dim=-1)

        return panel_encodings, all_att_weights, 0

    def forward(self, positions_batch, **kwargs):
        """3D to pattern with attention on per-point features"""

        batch_size = positions_batch.shape[0]

        # attention-based panel encodings
        panel_encodings, att_weights = self.forward_panel_enc_from_3d(positions_batch)

        # ---- decode panels from encodings ----

        panels = self.forward_panel_decode(panel_encodings.view(-1, panel_encodings.shape[-1]), batch_size)
        if len(att_weights) > 0:
            panels.update(att_weights=att_weights)  # save attention weights if non-empty

        return panels


class GarmentSegmentPattern3D(GarmentFullPattern3D):
    """
        Patterns from 3D data with point-level attention.
        Forward functions are subdivided for convenience of latent space inspection
    """
    def __init__(self, data_config, config={}, in_loss_config={}):

        if 'loss_components' not in in_loss_config:
            # with\wihtout attention losses!   , 'att_distribution', 'min_empty_att', 'stitch', 'free_class'
            in_loss_config.update(
                loss_components=['shape', 'loop', 'rotation', 'translation'], 
                quality_components=['shape', 'discrete', 'rotation', 'translation']
            )

        # training control defaults
        if 'freeze_on_clustering' not in config:
            config.update(freeze_on_clustering=False)

        super().__init__(data_config, config, in_loss_config)

        # set to true to get attention weights with prediction -- for visualization or loss evaluation
        # Keep false in all unnecessary cases to save memory!
        self.save_att_weights = (
            'att_distribution' in self.loss.config['loss_components'] 
            or 'segmentation' in self.loss.config['loss_components'])
        self.save_panel_enc = self.loss.config['cluster_by'] == 'panel_encodings'

        # defaults
        if 'local_attention' not in self.config:
            # Has to be false for the old runs that don't have this setting and rely on global attention
            self.config['local_attention'] = False  

        # ---- per-point attention module ---- 
        # that performs sort of segmentation
        # taking in per-point features and global encoding, outputting point weight per (potential) panel
        # Segmentaition aims to ensure that each point belongs to min number of panels
        # Global context gives understanding of the cutting pattern 
        attention_input_size = self.feature_extractor.config['EConv_feature']  
        if not self.config['local_attention']:  # adding global  feature
            attention_input_size += self.config['pattern_encoding_size']
        if self.config['skip_connections']:
            attention_input_size += 3  # initial coordinates

        self.point_segment_mlp = nn.Sequential(
            blocks.MLP([attention_input_size, attention_input_size, attention_input_size, self.max_pattern_size]),
            Sparsemax(dim=1)  # in the feature dimention
            # nn.Softmax(dim=1)   # DEBUG temporary solution for segmentation losses
        )

        # additional panel encoding post-procedding
        panel_att_out_size = self.feature_extractor.config['EConv_feature']
        if self.config['skip_connections']: 
            panel_att_out_size += 3
        self.panel_dec_lin = nn.Linear(
            panel_att_out_size, self.feature_extractor.config['panel_encoding_size'])

        # pattern decoder is not needed any more
        del self.pattern_decoder

    def forward_panel_enc_from_3d(self, positions_batch):
        """
            Get per-panel encodings from 3D data directly
            
        """
        # ------ Point cloud features -------
        batch_size = positions_batch.shape[0]
        # per-point and total encodings
        init_pattern_encodings, point_features_flat, batch = self.feature_extractor(
            positions_batch, 
            not self.config['local_attention']  # don't need global pool in this case
        )
        num_points = point_features_flat.shape[0] // batch_size

        # ----- Predict per-point panel scores (as attention weights) -----
        # propagate the per-pattern global encoding for each point
        if self.config['local_attention']:
            points_weights = self.point_segment_mlp(point_features_flat)
        else:
            global_enc_propagated = init_pattern_encodings.unsqueeze(1).repeat(1, num_points, 1).view(
                [-1, init_pattern_encodings.shape[-1]])

            points_weights = self.point_segment_mlp(torch.cat([global_enc_propagated, point_features_flat], dim=-1))

        # DEBUG 
        # print(f'Point Weights', points_weights)

        # ----- Getting per-panel features after attention application ------
        all_panel_features = []
        for panel_id in range(points_weights.shape[-1]):
            # get weights for particular panel
            panel_att_weights = points_weights[:, panel_id].unsqueeze(-1)

            # weight and pool to get panel encoding
            weighted_features = panel_att_weights * point_features_flat

            # same pool as in intial extractor
            panel_feature = self.feature_extractor.global_pool(weighted_features, batch, batch_size) 
            panel_feature = self.panel_dec_lin(panel_feature)  # reshape as needed
            panel_feature = panel_feature.view(batch_size, -1, panel_feature.shape[-1])

            all_panel_features.append(panel_feature)

        panel_encodings = torch.cat(all_panel_features, dim=1)  # concat in pattern dimention
        panel_encodings = panel_encodings.view(batch_size, -1, panel_encodings.shape[-1])

        points_weights = points_weights.view(batch_size, -1, points_weights.shape[-1]) if self.save_att_weights else []

        return panel_encodings, points_weights

    def forward(self, positions_batch, **kwargs):
        """3D to pattern with attention on per-point features"""

        batch_size = positions_batch.shape[0]

        epoch = kwargs['epoch'] if 'epoch' in kwargs else 0  # if not given -- go with default
        if self.config['freeze_on_clustering'] and epoch >= self.loss.config['epoch_with_cluster_checks']:
            self.freeze_panel_dec()
        elif self.training:  # avoid accidential freezing from evaluation mode propagated to training
            self.freeze_panel_dec(True)

        # attention-based panel encodings
        panel_encodings, att_weights = self.forward_panel_enc_from_3d(positions_batch)

        # ---- decode panels from encodings ----
        panels = self.forward_panel_decode(panel_encodings.view(-1, panel_encodings.shape[-1]), batch_size)

        if len(att_weights) > 0:
            panels.update(att_weights=att_weights)  # save attention weights if non-empty

        if self.save_panel_enc:
            panels.update(panel_encodings=panel_encodings)

        return panels

    def freeze_panel_dec(self, requires_grad=False):
        """ freeze parameters of panel_decoder """
        for param in self.panel_decoder.parameters():
            param.requires_grad = requires_grad


class GarmentSegment2EncPattern3D(GarmentFullPattern3D):
    """
        Patterns from 3D data with point-level attention. 
        Attention is computed by a separate encoder from the input point cloud
        Forward functions are subdivided for convenience of latent space inspection
    """
    def __init__(self, data_config, config={}, in_loss_config={}):

        if 'loss_components' not in in_loss_config:
            # with\wihtout attention losses!   , 'att_distribution', 'min_empty_att', 'stitch', 'free_class'
            in_loss_config.update(
                loss_components=['shape', 'loop', 'rotation', 'translation'], 
                quality_components=['shape', 'discrete', 'rotation', 'translation']
            )

        # training control defaults
        if 'freeze_on_clustering' not in config:
            config.update(freeze_on_clustering=False)

        super().__init__(data_config, config, in_loss_config)

        # set to true to get attention weights with prediction -- for visualization or loss evaluation
        # Keep false in all unnecessary cases to save memory!
        self.save_att_weights = 'att_distribution' in self.loss.config['loss_components'] or 'min_empty_att' in self.loss.config['loss_components']
        self.save_panel_enc = self.loss.config['cluster_by'] == 'panel_encodings'

        # defaults
        if 'local_attention' not in self.config:
            # Has to be false for the old runs that don't have this setting and rely on global attention
            self.config['local_attention'] = False  

        # ---- per-point attention module ---- 
        # that performs sort of segmentation
        # Uses encoder 
        # taking in per-point features and global encoding, outputting point weight per (potential) panel
        # Segmentaition aims to ensure that each point belongs to min number of panels
        # Global context gives understanding of the cutting pattern 

        feature_extractor_module = getattr(blocks, self.config['feature_extractor'])
        self.attention_extractor = feature_extractor_module(self.config['pattern_encoding_size'], self.config)
        # DEBUG
        # if hasattr(self.attention_extractor, 'config'):
        #    self.config.update(self.feature_extractor.config)   # save extractor's additional configuration

        # TODO Maybe just apply one layer & SparseMax?  
        attention_input_size = self.attention_extractor.config['EConv_feature']  
        if not self.config['local_attention']:  # adding global  feature
            attention_input_size += self.config['pattern_encoding_size']
        if self.config['skip_connections']:
            attention_input_size += 3  # initial coordinates
        self.point_segment_mlp = nn.Sequential(
            blocks.MLP([attention_input_size, attention_input_size, attention_input_size, self.max_pattern_size]),
            Sparsemax(dim=1)  # in the feature dimention
        )

        # additional panel encoding post-procedding
        panel_att_out_size = self.feature_extractor.config['EConv_feature']
        if self.config['skip_connections']: 
            panel_att_out_size += 3
        self.panel_dec_lin = nn.Linear(
            panel_att_out_size, self.feature_extractor.config['panel_encoding_size'])

        # pattern decoder is not needed any more
        del self.pattern_decoder

    def forward_panel_enc_from_3d(self, positions_batch):
        """
            Get per-panel encodings from 3D data directly
            
        """
        # ------ Point cloud features -------
        batch_size = positions_batch.shape[0]
        # per-point and total encodings
        init_pattern_encodings, _, batch = self.feature_extractor(
            positions_batch, 
            not self.config['local_attention']  # don't need global pool in this case
        )
        

        # ----- Predict per-point panel scores (as attention weights) -----
        # get encodings for attention feature for each point
        _, point_features_flat, batch = self.attention_extractor(
            positions_batch, 
            not self.config['local_attention']  # don't need global pool in this case
        )
        num_points = point_features_flat.shape[0] // batch_size

        # use the features to predict attention points
        if self.config['local_attention']:
            points_weights = self.point_segment_mlp(point_features_flat)
        else:
            global_enc_propagated = init_pattern_encodings.unsqueeze(1).repeat(1, num_points, 1).view(
                [-1, init_pattern_encodings.shape[-1]])

            points_weights = self.point_segment_mlp(torch.cat([global_enc_propagated, point_features_flat], dim=-1))

        # ----- Getting per-panel features after attention application ------
        all_panel_features = []
        for panel_id in range(points_weights.shape[-1]):
            # get weights for particular panel
            panel_att_weights = points_weights[:, panel_id].unsqueeze(-1)

            # weight and pool to get panel encoding
            weighted_features = panel_att_weights * point_features_flat

            # same pool as in intial extractor
            panel_feature = self.feature_extractor.global_pool(weighted_features, batch, batch_size) 
            panel_feature = self.panel_dec_lin(panel_feature)  # reshape as needed
            panel_feature = panel_feature.view(batch_size, -1, panel_feature.shape[-1])

            all_panel_features.append(panel_feature)

        panel_encodings = torch.cat(all_panel_features, dim=1)  # concat in pattern dimention
        panel_encodings = panel_encodings.view(batch_size, -1, panel_encodings.shape[-1])

        points_weights = points_weights.view(batch_size, -1, points_weights.shape[-1]) if self.save_att_weights else []

        return panel_encodings, points_weights

    def forward(self, positions_batch, **kwargs):
        """3D to pattern with attention on per-point features"""

        batch_size = positions_batch.shape[0]

        epoch = kwargs['epoch'] if 'epoch' in kwargs else 0  # if not given -- go with default
        if self.config['freeze_on_clustering'] and epoch >= self.loss.config['epoch_with_cluster_checks']:
            self.freeze_panel_dec()
        elif self.training:  # avoid accidential freezing from evaluation mode propagated to training
            self.freeze_panel_dec(True)

        # attention-based panel encodings
        panel_encodings, att_weights = self.forward_panel_enc_from_3d(positions_batch)

        # ---- decode panels from encodings ----
        panels = self.forward_panel_decode(panel_encodings.view(-1, panel_encodings.shape[-1]), batch_size)

        if len(att_weights) > 0:
            panels.update(att_weights=att_weights)  # save attention weights if non-empty

        if self.save_panel_enc:
            panels.update(panel_encodings=panel_encodings)

        return panels

    def freeze_panel_dec(self, requires_grad=False):
        """ freeze parameters of panel_decoder """
        for param in self.panel_decoder.parameters():
            param.requires_grad = requires_grad



# ----------- Stitches (independent) predictions ---------
class StitchOnEdge3DPairs(BaseModule):
    """
        Predicting status of a particular pair of edges (defined in 3D) -- whether they are connected 
        with a stitch or not.

        Binary classification problem
    """

    def __init__(self, data_config, config={}, in_loss_config={}):
        super().__init__()

        # data props
        self.pair_feature_len = data_config['element_size']

        # ---- Net configuration ----
        self.config.update({
            'stitch_hidden_size': 20, 
            'stitch_mlp_n_layers': 3
        })
        # update with input settings
        self.config.update(config) 

        # --- Losses ---
        self.config['loss'] = {
            'loss_components': ['edge_pair_class'],
            'quality_components': ['edge_pair_class', 'edge_pair_stitch_recall'],
            'panel_origin_invariant_loss': False,  # don't even try to evaluate
            'panel_order_inariant_loss': False
        }
        self.config['loss'].update(in_loss_config)  # apply input settings 

        # create loss!
        self.loss = ComposedLoss(data_config, self.config['loss'])
        self.config['loss'] = self.loss.config  # sync

        # ------ Modules ----
        mid_layers = [self.config['stitch_hidden_size']] * self.config['stitch_mlp_n_layers']
        self.mlp = blocks.MLP([self.pair_feature_len] + mid_layers + [1])


    def forward(self, pairs_batch, **kwargs):
        self.device = pairs_batch.device
        self.batch_size = pairs_batch.size(0)
        return_shape = list(pairs_batch.shape)
        return_shape.pop(-1)

        # reduce extra dimentions if needed
        out = self.mlp(pairs_batch.contiguous().view(-1, pairs_batch.shape[-1])) 

        # follow the same dimentions structure
        return out.view(return_shape)



if __name__ == "__main__":

    torch.manual_seed(125)

    a = torch.arange(1, 25, dtype=torch.float)
    dataset_gt = a.view(-1, 2, 3)
    # print(dataset_gt)
    gt_batch = a.view(2, -1, 2, 3)  # ~ 2 examples in batch
    # print(gt_batch)
    net = GarmentFullPattern3D(
        gt_batch.shape[3], gt_batch.shape[2], gt_batch.shape[1], 6, 3)  # {'shift': dataset_gt.mean(), 'scale': dataset_gt.std()})

    positions = torch.arange(1, 37, dtype=torch.float)
    features_batch = positions.view(2, -1, 3)  # note for the same batch size

    print('In batch shape: {}; Out batch shape: {}'.format(features_batch.shape, gt_batch.shape))
    print(net(features_batch)) 
    loss = net.loss(features_batch, gt_batch)
    print(loss)
    loss.backward()  # check it doesn't fail
