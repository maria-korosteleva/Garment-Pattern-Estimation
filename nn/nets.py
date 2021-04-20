import torch
import torch.nn as nn
import torch.nn.functional as F

# my modules
import metrics
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


# -------- Nets architectures -----------
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
        self.loss = metrics.ComposedPatternLoss(data_config, self.config['loss'])
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

    def forward(self, patterns_batch):
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

    def forward(self, patterns_batch):
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
            'panel_n_layers': 4, 
            'pattern_encoding_size': 130, 
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
            'loss_components': ['shape', 'loop', 'rotation', 'translation', 'stitch', 'free_class'],
            'quality_components': ['shape', 'discrete', 'rotation', 'translation', 'stitch', 'free_class'],
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
        self.loss = metrics.ComposedPatternLoss(data_config, self.config['loss'])
        self.config['loss'] = self.loss.config  # sync just in case

        # ---- Feature extractor definition -------
        feature_extractor_module = getattr(blocks, self.config['feature_extractor'])
        self.feature_extractor = feature_extractor_module(self.config['pattern_encoding_size'], self.config)
        if hasattr(self.feature_extractor, 'config'):
            self.config.update(self.feature_extractor.config)   # save extractor's additional configuration

        # ----- Decode into pattern definition -------
        panel_decoder_module = getattr(blocks, self.config['panel_decoder'])
        self.panel_decoder = panel_decoder_module(
            self.config['panel_encoding_size'], self.config['panel_encoding_size'], 
            self.panel_elem_len + self.config['stitch_tag_dim'] + 1,  # last element is free tag indicator 
            self.config['panel_n_layers'], 
            dropout=self.config['dropout'], 
            custom_init=self.config['lstm_init']
        )
        pattern_decoder_module = getattr(blocks, self.config['pattern_decoder'])
        self.pattern_decoder = pattern_decoder_module(
            self.config['pattern_encoding_size'], self.config['pattern_encoding_size'], self.config['panel_encoding_size'], self.config['pattern_n_layers'], 
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
        return self.feature_extractor(positions_batch)  # YAAAAY Pattern hidden representation!!

    def forward_pattern_decode(self, garment_encodings):
        """
            Unfold provided garment encodings into per-panel encodings
            Useful for obtaining the latent space for Panels
        """
        panel_encodings = self.pattern_decoder(garment_encodings, self.max_pattern_size)
        flat_panel_encodings = panel_encodings.contiguous().view(-1, panel_encodings.shape[-1])

        return flat_panel_encodings

    def forward_decode(self, garment_encodings):
        """
            Unfold provided garment encodings into the sewing pattens
        """
        self.device = garment_encodings.device
        batch_size = garment_encodings.size(0)

        flat_panel_encodings = self.forward_pattern_decode(garment_encodings)

        # Panel outlines & stitch info
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

    def forward(self, positions_batch):
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
