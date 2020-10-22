import torch
import torch.nn as nn
import torch.nn.functional as F

# my modules
import metrics
import net_blocks as blocks


# ------ Basic Interface --------
class BaseModule(nn.Module):
    """Base interface for my neural nets"""
    def __init__(self):
        super().__init__()
        self.config = {'loss': 'MSELoss'}
        self.regression_loss = nn.MSELoss()
    
    def loss(self, features, ground_truth):
        """Default loss for my neural networks. Takes pne batch of data"""
        preds = self(features)
        ground_truth = ground_truth.to(features.device)  # make sure device is correct
        loss = self.regression_loss(preds, ground_truth)
        return loss, {'regression loss': loss}  # second term is for compound losses


# -------- Nets architectures -----------

class ShirtfeaturesMLP(BaseModule):
    """MLP for training on shirts dataset. Assumes 100 features parameters used"""
    
    def __init__(self, in_size, out_size):
        super().__init__()
        
        # layers definitions
        self.sequence = nn.Sequential(
            nn.Linear(in_size, 300),  # nn.Linear(36756, 3000),
            nn.ReLU(), 
            nn.Linear(300, 300),  # nn.Linear(3000, 300)
            nn.ReLU(), 
            nn.Linear(300, 60),
            nn.ReLU(),
            nn.Linear(60, out_size)
        )
    
    def forward(self, x_batch):
        # print (x_batch)
        
        return self.sequence(x_batch)


class GarmentParamsMLP(BaseModule):
    """MLP for training on shirts dataset. Assumes 100 features parameters used"""
    
    def __init__(self, in_size, out_size):
        super().__init__()
        
        # layers definitions
        self.sequence = nn.Sequential(
            nn.Linear(in_size, 300),
            nn.ReLU(), 
            nn.Linear(300, 300),
            nn.ReLU(), 
            nn.Linear(300, 60),
            nn.ReLU(),
            nn.Linear(60, out_size)
        )
    
    def forward(self, x_batch):
        # print (x_batch)
        
        return self.sequence(x_batch)


class GarmentParamsPoint(BaseModule):
    """PointNet++ processing of input geometry to predict parameters
        Note that architecture is agnostic of number of input points"""
    def __init__(self, out_size, config={}):
        super().__init__()

        self.config.update({'r1': 10, 'r2': 40})  # defaults for this net
        self.config.update(config)  # from input

        self.feature_extractor = blocks.PointNetPlusPlus(512, {'r1': self.config['r1'], 'r2': self.config['r2']})

        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, out_size)

    def forward(self, positions):

        out = self.feature_extractor(positions)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = F.relu(self.lin2(out))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.lin3(out)
        return out


class GarmentPanelsAE(BaseModule):
    """Model for sequential encoding & decoding of garment panels
        References: 
        * https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
        * for seq2seq decisions https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/
    """
    def __init__(self, in_elem_len, max_seq_len, data_norm={}, config={}):
        super().__init__()

        # defaults for this net
        self.config.update({
            'hidden_dim_enc': 20, 
            'hidden_dim_dec': 20, 
            'n_layers': 3, 
            'loop_loss_weight': 0.1, 
            'dropout': 0.1,
            'lstm_init': 'kaiming_normal_',
            'decoder': 'LSTMDoubleReverseDecoderModule'
        })
        # update with input settings
        self.config.update(config) 

        # additional info
        self.config['loss'] = 'MSE Reconstruction with loop'

        self.loop_loss = metrics.PanelLoopLoss(data_stats=data_norm)

        # encode
        self.seq_encoder = blocks.LSTMEncoderModule(
            in_elem_len, self.config['hidden_dim_enc'], self.config['n_layers'], dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )

        # decode
        if self.config['decoder'] == 'LSTMDecoderModule':
            self.seq_decoder = blocks.LSTMDecoderModule(
                self.config['hidden_dim_enc'], self.config['hidden_dim_dec'], in_elem_len, self.config['n_layers'], 
                dropout=self.config['dropout'],
                custom_init=self.config['lstm_init']
            )
        elif self.config['decoder'] == 'LSTMDoubleReverseDecoderModule':
            self.seq_decoder = blocks.LSTMDoubleReverseDecoderModule(
                self.config['hidden_dim_enc'], self.config['hidden_dim_dec'], in_elem_len, self.config['n_layers'], 
                dropout=self.config['dropout'],
                custom_init=self.config['lstm_init']
            )
        else:
            raise ValueError('GarmentPattern3D::Error::Unsupported decoder {} requested in config'.format(self.config['decoder']))
        

    def forward(self, x):

        encoding = self.seq_encoder(x)  # Yay

        out = self.seq_decoder(encoding, x.shape[-2])  # -2 corresponds to len of paddel panel
        return out

    def loss(self, features, ground_truth):
        """Override base class loss calculation to use reconstruction loss"""
        preds = self(features)

        # Base reconstruction loss
        reconstruction_loss = self.regression_loss(preds, features)   # features are the ground truth in this case -> reconstruction loss
        # ensuring edges within panel loop & return to origin
        loop_loss = self.loop_loss(preds, features)

        return reconstruction_loss + self.config['loop_loss_weight'] * loop_loss


class GarmentPatternAE(BaseModule):
    """
        Model to test hierarchical encoding & decoding of garment 2D patterns (as panel collection)
        Based on findings from GarmentPanelsAE
    """
    def __init__(self, in_elem_len, max_panel_len, data_norm={}, config={}):
        super().__init__()

        # defaults for this net
        self.config.update({
            'panel_encoding_size': 20, 
            'panel_n_layers': 3, 
            'pattern_encoding_size': 40, 
            'pattern_n_layers': 3, 
            'loop_loss_weight': 0.1, 
            'dropout': 0,
            'lstm_init': 'kaiming_normal_', 
            'decoder': 'LSTMDoubleReverseDecoderModule'
        })
        # update with input settings
        self.config.update(config) 

        # additional info
        self.config['loss'] = 'MSE Reconstruction with loop'

        self.loop_loss = metrics.PanelLoopLoss(data_stats=data_norm)

        decoder_module = getattr(blocks, self.config['decoder'])

        # --- panel-level ---- 
        self.panel_encoder = blocks.LSTMEncoderModule(
            in_elem_len, self.config['panel_encoding_size'], self.config['panel_n_layers'], 
            dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )
        self.panel_decoder = blocks.LSTMDecoderModule(
            self.config['panel_encoding_size'], self.config['panel_encoding_size'], in_elem_len, self.config['panel_n_layers'], 
            dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )

        # ----- patten level ------
        self.pattern_encoder = blocks.LSTMEncoderModule(
            self.config['panel_encoding_size'], self.config['pattern_encoding_size'], self.config['pattern_n_layers'], 
            dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )
        self.pattern_decoder = decoder_module(
            self.config['pattern_encoding_size'], self.config['pattern_encoding_size'], self.config['panel_encoding_size'], self.config['pattern_n_layers'], 
            dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )

    def forward(self, patterns_batch):
        self.device = patterns_batch.device
        batch_size = patterns_batch.size(0)
        pattern_size = patterns_batch.size(1)
        panel_size = patterns_batch.size(2)

        # --- Encode ---
        # flatten -- view simply as a list of panels to apply encoding per panel
        all_panels = patterns_batch.contiguous().view(-1, patterns_batch.shape[-2], patterns_batch.shape[-1])
        panel_encodings = self.panel_encoder(all_panels)

        panel_encodings = panel_encodings.contiguous().view(batch_size, pattern_size, -1)  # group by patterns
        pattern_encoding = self.pattern_encoder(panel_encodings)   # YAAAAY Pattern hidden representation!!

        # --- Decode ---
        panel_encodings_dec = self.pattern_decoder(pattern_encoding, pattern_size)

        flat_panel_encodings_dec = panel_encodings_dec.contiguous().view(-1, panel_encodings_dec.shape[-1])
        flat_panels_dec = self.panel_decoder(flat_panel_encodings_dec, panel_size)

        # back to patterns and panels
        prediction = flat_panels_dec.contiguous().view(batch_size, pattern_size, panel_size, -1)
        
        return prediction

    def loss(self, features, ground_truth):
        """Override base class loss calculation to use reconstruction loss"""
        preds = self(features)

        # ---- Base reconstruction loss -----
        reconstruction_loss = self.regression_loss(preds, features)   # features are the ground truth in this case -> reconstruction loss

        # ---- Loop loss -----
        loop_loss = self.loop_loss(preds, features)

        # return format
        loss_dict = dict(pattern_loss=reconstruction_loss, loop_loss=loop_loss)

        return reconstruction_loss + self.config['loop_loss_weight'] * loop_loss, loss_dict


class GarmentPattern3D(BaseModule):
    """
        Predicting 2D pattern geometry (panels outlines) from 3D garment geometry 
        Constists of (interchangeable) feature extractor and pattern decoder from GarmentPatternAE
    """
    def __init__(self, panel_elem_len, max_panel_len, max_pattern_size, data_norm={}, config={}):
        super().__init__()

        # defaults for this net
        self.config.update({
            'panel_encoding_size': 70, 
            'panel_n_layers': 4, 
            'pattern_encoding_size': 130, 
            'pattern_n_layers': 3, 
            'loop_loss_weight': 0.1, 
            'dropout': 0,
            'loss': 'MSE with loop',
            'lstm_init': 'kaiming_normal_', 
            'feature_extractor': 'EdgeConvFeatures',
            'panel_decoder': 'LSTMDecoderModule', 
            'pattern_decoder': 'LSTMDecoderModule'
        })
        # update with input settings
        self.config.update(config) 

        # output props
        self.max_panel_len = max_panel_len
        self.max_pattern_size = max_pattern_size

        # extra loss object
        self.loop_loss = metrics.PanelLoopLoss(data_stats=data_norm)

        # Feature extractor definition
        feature_extractor_module = getattr(blocks, self.config['feature_extractor'])
        self.feature_extractor = feature_extractor_module(self.config['pattern_encoding_size'], self.config)
        if hasattr(self.feature_extractor, 'config'):
            self.config.update(self.feature_extractor.config)   # save extractor's additional configuration


        # Decode into pattern definition
        panel_decoder_module = getattr(blocks, self.config['panel_decoder'])
        self.panel_decoder = panel_decoder_module(
            self.config['panel_encoding_size'], self.config['panel_encoding_size'], panel_elem_len, self.config['panel_n_layers'], 
            dropout=self.config['dropout'], 
            custom_init=self.config['lstm_init']
        )
        pattern_decoder_module = getattr(blocks, self.config['pattern_decoder'])
        self.pattern_decoder = pattern_decoder_module(
            self.config['pattern_encoding_size'], self.config['pattern_encoding_size'], self.config['panel_encoding_size'], self.config['pattern_n_layers'], 
            dropout=self.config['dropout'],
            custom_init=self.config['lstm_init']
        )

    def forward(self, positions_batch):
        self.device = positions_batch.device
        batch_size = positions_batch.size(0)

        # Extract info from geometry 
        pattern_encoding = self.feature_extractor(positions_batch)  # YAAAAY Pattern hidden representation!!

        # Decode 
        panel_encodings = self.pattern_decoder(pattern_encoding, self.max_pattern_size)

        flat_panel_encodings = panel_encodings.contiguous().view(-1, panel_encodings.shape[-1])
        flat_panels = self.panel_decoder(flat_panel_encodings, self.max_panel_len)

        # back to patterns and panels
        prediction = flat_panels.contiguous().view(batch_size, self.max_pattern_size, self.max_panel_len, -1)
        
        return prediction

    def loss(self, features, ground_truth):
        """Evalute loss when predicting patterns"""
        preds = self(features)
        ground_truth = ground_truth.to(features.device)  # make sure device is correct

        # Base extraction loss 
        pattern_loss = self.regression_loss(preds, ground_truth) 

        # Loop loss per panel
        loop_loss = self.loop_loss(preds, ground_truth)

        # return format
        loss_dict = dict(pattern_loss=pattern_loss, loop_loss=loop_loss)

        return pattern_loss + self.config['loop_loss_weight'] * loop_loss, loss_dict


class GarmentFullPattern3D(BaseModule):
    """
        Predicting 2D pattern inluding panel placement and stitches information from 3D garment geometry 
        Constists of 
            * (interchangeable) feature extractor 
            * pattern decoder from GarmentPatternAE
            * MLP modules to predict panel 3D placement & stitches
    """
    def __init__(self, panel_elem_len, max_panel_len, max_pattern_size, rotation_size, translation_size, data_norm={}, config={}):
        super().__init__()

        # defaults for this net
        self.config.update({
            'panel_encoding_size': 70, 
            'panel_n_layers': 4, 
            'pattern_encoding_size': 130, 
            'pattern_n_layers': 3, 
            'loop_loss_weight': 0.1, 
            'dropout': 0,
            'loss': 'MSE with loop',
            'lstm_init': 'kaiming_normal_', 
            'feature_extractor': 'EdgeConvFeatures',
            'panel_decoder': 'LSTMDecoderModule', 
            'pattern_decoder': 'LSTMDecoderModule'
        })
        # update with input settings
        self.config.update(config) 

        # output props
        self.max_panel_len = max_panel_len
        self.max_pattern_size = max_pattern_size
        self.rotation_size = rotation_size
        self.translation_size = translation_size

        # extra loss object
        self.loop_loss = metrics.PanelLoopLoss(
            data_stats={'mean': data_norm['gt_mean']['outlines'], 'std': data_norm['gt_std']['outlines']})

        # Feature extractor definition
        feature_extractor_module = getattr(blocks, self.config['feature_extractor'])
        self.feature_extractor = feature_extractor_module(self.config['pattern_encoding_size'], self.config)
        if hasattr(self.feature_extractor, 'config'):
            self.config.update(self.feature_extractor.config)   # save extractor's additional configuration

        # Decode into pattern definition
        panel_decoder_module = getattr(blocks, self.config['panel_decoder'])
        self.panel_decoder = panel_decoder_module(
            self.config['panel_encoding_size'], self.config['panel_encoding_size'], panel_elem_len, self.config['panel_n_layers'], 
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
        self.rotation_decoder = blocks.MLP([
            self.config['panel_encoding_size'],
            100,
            rotation_size
        ])
        self.translation_decoder = blocks.MLP([
            self.config['panel_encoding_size'],
            100,
            translation_size
        ])

        # TODO add stitches prediction modules

    def forward(self, positions_batch):
        self.device = positions_batch.device
        batch_size = positions_batch.size(0)

        # Extract info from geometry 
        pattern_encoding = self.feature_extractor(positions_batch)  # YAAAAY Pattern hidden representation!!

        # Decode 
        panel_encodings = self.pattern_decoder(pattern_encoding, self.max_pattern_size)
        flat_panel_encodings = panel_encodings.contiguous().view(-1, panel_encodings.shape[-1])

        # Panel outlines
        flat_panels = self.panel_decoder(flat_panel_encodings, self.max_panel_len)
        
        # Placement
        flat_rotations = self.rotation_decoder(flat_panel_encodings) 
        flat_translations = self.translation_decoder(flat_panel_encodings)

        # reshape to per-pattern predictions
        outlines = flat_panels.contiguous().view(batch_size, self.max_pattern_size, self.max_panel_len, -1)
        rotations = flat_rotations.contiguous().view(batch_size, self.max_pattern_size, -1)
        translations = flat_translations.contiguous().view(batch_size, self.max_pattern_size, -1)

        return {'outlines': outlines, 'rotations': rotations, 'translations': translations}

    def loss(self, features, ground_truth):
        """Evalute loss when predicting patterns"""
        preds = self(features)
        device = features.device

        # Loss for panel shapes
        outlines = ground_truth['outlines'].to(device)
        pattern_loss = self.regression_loss(preds['outlines'], outlines)   
        # Loop loss per panel
        loop_loss = self.loop_loss(preds['outlines'], outlines)

        # panel placement
        rot_loss = self.regression_loss(preds['rotations'], ground_truth['rotations'].to(device))
        translation_loss = self.regression_loss(preds['translations'], ground_truth['translations'].to(device))

        loss_dict = dict(
            pattern_loss=pattern_loss, loop_loss=loop_loss, 
            rotation_loss=rot_loss, translation_loss=translation_loss)

        return pattern_loss + self.config['loop_loss_weight'] * loop_loss + rot_loss + translation_loss, loss_dict




if __name__ == "__main__":

    torch.manual_seed(125)

    a = torch.arange(1, 25, dtype=torch.float)
    dataset_gt = a.view(-1, 2, 3)
    # print(dataset_gt)
    gt_batch = a.view(2, -1, 2, 3)  # ~ 2 examples in batch
    # print(gt_batch)
    net = GarmentFullPattern3D(
        gt_batch.shape[3], gt_batch.shape[2], gt_batch.shape[1], 6, 3)  # {'mean': dataset_gt.mean(), 'std': dataset_gt.std()})

    positions = torch.arange(1, 37, dtype=torch.float)
    features_batch = positions.view(2, -1, 3)  # note for the same batch size

    print('In batch shape: {}; Out batch shape: {}'.format(features_batch.shape, gt_batch.shape))
    print(net(features_batch)) 
    loss = net.loss(features_batch, gt_batch)
    print(loss)
    loss.backward()  # check it doesn't fail
