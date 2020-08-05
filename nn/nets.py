import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geometric

class BaseModule(nn.Module):
    """Base interface for my neural nets"""
    def __init__(self):
        super().__init__()
        self.config = {'loss': 'MSELoss'}
        self.regression_loss = nn.MSELoss()
    
    def loss(self, features, ground_truth):
        """Default loss for my neural networks. Takes pne batch of data"""
        preds = self(features)
        return self.regression_loss(preds, ground_truth)


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
        #print (x_batch)
        
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
        #print (x_batch)
        
        return self.sequence(x_batch)


# --------- PointNet++ - based -------
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py

class SetAbstractionModule(nn.Module):
    """Performes PointNet feature extraction in local areas (around sampled centroids) of given sets"""
    def __init__(self, ratio, conv_radius, per_point_nn):
        super().__init__()
        self.ratio = ratio  # for controlling number of centroids
        self.radius = conv_radius
        self.conv = geometric.PointConv(per_point_nn)

    def forward(self, features, pos, batch):
        idx = geometric.fps(pos, batch, ratio=self.ratio)
        row, col = geometric.radius(pos, pos[idx], self.radius, batch, batch[idx],
                            max_num_neighbors=32)
        edge_index = torch.stack([col, row], dim=0)
        features = self.conv(features, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return features, pos, batch


class GlobalSetAbstractionModule(nn.Module):
    """PointNet feature extraction to get one deature vector from set"""
    def __init__(self, per_point_net):
        super().__init__()
        self.nn = per_point_net

    def forward(self, features, pos, batch):
        features = self.nn(torch.cat([features, pos], dim=1))
        features = geometric.global_max_pool(features, batch)  # returns classical PyTorch batch format #Batch_size x (out_shape)
        pos = pos.new_zeros((features.size(0), 3))
        batch = torch.arange(features.size(0), device=batch.device)
        return features, pos, batch


def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])


class GarmentParamsPoint(BaseModule):
    """PointNet++ processing of input geometry to predict parameters
        Note that architecture is agnostic of number of input points"""
    def __init__(self, out_size, config={}):
        super().__init__()

        self.config.update({'r1': 10, 'r2': 40})  # defaults for this net
        self.config.update(config)  # from input

        self.sa1_module = SetAbstractionModule(0.5, config['r1'], MLP([3, 64, 64, 128]))
        self.sa2_module = SetAbstractionModule(0.25, config['r2'], MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSetAbstractionModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, out_size)

    def forward(self, positions):

        # flatten the batch for torch-geometric batch format
        pos_flat = positions.view(-1, positions.size(-1))
        batch = torch.cat([
            torch.full((elem.size(0),), fill_value=i, device=positions.device, dtype=torch.long) for i, elem in enumerate(positions)
        ])

        # forward pass
        sa0_out = (None, pos_flat, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = sa3_out
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x


# ------------- Pattern representation ----

def init_tenzor(*shape, device='cpu'):
    """shortcut to create & initialize tenzors on a given device.  """
    new_tenzor = torch.empty(shape)
    nn.init.kaiming_normal_(new_tenzor)  # TODO suport other init types 
    return new_tenzor.to(device)

# --- Util modules ---
class LSTMEncoderModule(nn.Module):
    """A wrapper for LSTM targeting encoding task"""
    def __init__(self, elem_len, encoding_size, n_layers, dropout=0):
        super().__init__()
        self.n_layers = n_layers
        self.encoding_size = encoding_size
        self.lstm = nn.LSTM(
            elem_len, encoding_size, n_layers, 
            dropout=dropout, batch_first=True)

    def forward(self, batch_sequence):
        device = batch_sequence.device
        batch_size = batch_sequence.size(0)
        
        # --- encode --- 
        hidden_init = init_tenzor(self.n_layers, batch_size, self.encoding_size, device=device)
        cell_init = init_tenzor(self.n_layers, batch_size, self.encoding_size, device=device)
        _, (hidden, _) = self.lstm(batch_sequence, (hidden_init, cell_init))

        # final encoding is the last output == hidden of last layer 
        return hidden[-1]


class LSTMDecoderModule(nn.Module):
    """A wrapper for LSTM targeting decoding task"""
    def __init__(self, encoding_size, hidden_size, out_elem_size, n_layers, dropout=0):
        super().__init__()
        self.n_layers = n_layers
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.out_elem_size = out_elem_size

        self.lstm = nn.LSTM(encoding_size, hidden_size, n_layers, 
                            dropout=dropout, batch_first=True)

        # post-process to match the desired outut shape
        self.lin = nn.Linear(hidden_size, out_elem_size)

    def forward(self, batch_enc, out_len):
        """out_len specifies the length of the output sequence to produce"""
        device = batch_enc.device
        batch_size = batch_enc.size(0)
        
        # propagate encoding for needed seq_len
        dec_input = batch_enc.unsqueeze(1).repeat(1, out_len, 1)  # along sequence dimention

        # decode
        hidden_init = init_tenzor(self.n_layers, batch_size, self.hidden_size, device=device)
        cell_init = init_tenzor(self.n_layers, batch_size, self.hidden_size, device=device)
        out, _ = self.lstm(dec_input, (hidden_init, cell_init))
        
        # back to requested format
        # reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.lin(out)
        # back to sequence
        out = out.contiguous().view(batch_size, out_len, -1)

        return out


# --- Nets classes ---
class GarmentPanelsAE(BaseModule):
    """Model for sequential encoding & decoding of garment panels
        References: 
        * https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
        * for seq2seq decisions https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/
    """
    def __init__(self, in_elem_len, max_seq_len, data_norm={}, config={}):
        super().__init__()

        # defaults for this net
        self.config.update({'hidden_dim_enc': 20, 'hidden_dim_dec': 20, 'n_layers': 3, 'loop_loss_weight': 0.1, 'dropout': 0.1})
        # update with input settings
        self.config.update(config) 

        # additional info
        self.config['loss'] = 'MSE Reconstruction with loop'
        self.config['hidden_init'] = 'kaiming_normal_'

        self.pad_tenzor = -data_norm['mean'] / data_norm['std'] if data_norm else torch.zeros(in_elem_len)
        if not torch.is_tensor(self.pad_tenzor):
            self.pad_tenzor = torch.Tensor(self.pad_tenzor)
        self.pad_tenzor = self.pad_tenzor.repeat(max_seq_len, 1)

        # encode
        self.seq_encoder = LSTMEncoderModule(
            in_elem_len, self.config['hidden_dim_enc'], self.config['n_layers'], dropout=self.config['dropout']
        )

        # decode
        self.seq_decoder = LSTMDecoderModule(
            self.config['hidden_dim_enc'], self.config['hidden_dim_dec'], in_elem_len, self.config['n_layers'], 
            dropout=self.config['dropout']
        )

        # init values
        self.init_net_params()

    def forward(self, x):

        encoding = self.seq_encoder(x)  # Yay

        out = self.seq_decoder(encoding, x.shape[-2])  # -2 corresponds to len of paddel panel
        return out

    def init_net_params(self):
        """Apply custom initialization to net parameters"""
        self.config['init'] = 'kaiming_normal_'
        for name, param in self.seq_encoder.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            # leave defaults for bias
        for name, param in self.seq_decoder.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            # leave defaults for bias

    def loss(self, features, ground_truth):
        """Override base class loss calculation to use reconstruction loss"""
        preds = self(features)

        # Base reconstruction loss
        reconstruction_loss = self.regression_loss(preds, features)   # features are the ground truth in this case -> reconstruction loss

        # ensuring edges within panel loop & return to origin
        panel_coords_sum = torch.zeros((features.shape[0], 2))
        panel_coords_sum = panel_coords_sum.to(device=features.device)
        self.pad_tenzor = self.pad_tenzor.to(device=features.device)
        for el_id in range(features.shape[0]):
            # iterate over elements in batch
            # loop loss per panel + we need to know each panel original length
            panel = features[el_id]
            # unpad
            bool_matrix = torch.isclose(panel, self.pad_tenzor, atol=1.e-2)
            seq_len = (~torch.all(bool_matrix, axis=1)).sum()  # only non-padded rows

            # update loss
            panel_coords_sum[el_id] = preds[el_id][:seq_len, :2].sum(axis=0)

        # panel_coords_sum = preds.sum(axis=1)[:, 0:2]  # taking only edge vectors' endpoints -- ignoring curvature coords
        panel_square_sums = panel_coords_sum ** 2  # per sum square

        # batch mean of squared norms of per-panel final points:
        loop_loss = panel_square_sums.sum() / panel_square_sums.shape[0]

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
            'dropout': 0})
        # update with input settings
        self.config.update(config) 

        # additional info
        self.config['loss'] = 'MSE Reconstruction with loop'
        self.config['hidden_init'] = 'kaiming_normal_'

        self.pad_tenzor = -data_norm['mean'] / data_norm['std'] if data_norm else torch.zeros(in_elem_len)
        if not torch.is_tensor(self.pad_tenzor):
            self.pad_tenzor = torch.Tensor(self.pad_tenzor)
        self.pad_tenzor = self.pad_tenzor.repeat(max_panel_len, 1)

        # --- panel-level ---- 
        self.panel_encoder = LSTMEncoderModule(
            in_elem_len, self.config['panel_encoding_size'], self.config['panel_n_layers'], 
            dropout=self.config['dropout']
        )
        self.panel_decoder = LSTMDecoderModule(
            self.config['panel_encoding_size'], self.config['panel_encoding_size'], in_elem_len, self.config['panel_n_layers'], 
            dropout=self.config['dropout']
        )

        # ----- patten level ------
        self.pattern_encoder = LSTMEncoderModule(
            self.config['panel_encoding_size'], self.config['pattern_encoding_size'], self.config['pattern_n_layers'], 
            dropout=self.config['dropout']
        )
        self.pattern_decoder = LSTMDecoderModule(
            self.config['pattern_encoding_size'], self.config['pattern_encoding_size'], self.config['panel_encoding_size'], self.config['pattern_n_layers'], 
            dropout=self.config['dropout']
        )

        # init values
        self.init_submodule_params(self.panel_encoder)
        self.init_submodule_params(self.panel_decoder)
        self.init_submodule_params(self.pattern_encoder)
        self.init_submodule_params(self.pattern_decoder)

    def forward(self, patterns_batch):
        self.device = patterns_batch.device
        batch_size = patterns_batch.size(0)
        pattern_size = patterns_batch.size(1)
        panel_size = patterns_batch.size(2)
        num_panels = batch_size * pattern_size

        # --- Encode ---
        # flatten -- view simply as a list of panels to apply encoding per panel
        all_panels = patterns_batch.contiguous().view(num_panels, patterns_batch.shape[-2], patterns_batch.shape[-1])
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

    def init_submodule_params(self, submodule):
        """Apply custom initialization to net parameters of given submodule"""
        self.config['init'] = 'kaiming_normal_'
        for name, param in submodule.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            # leave defaults for bias

    def loss(self, features, ground_truth):
        """Override base class loss calculation to use reconstruction loss"""
        preds = self(features)

        # ---- Base reconstruction loss -----
        reconstruction_loss = self.regression_loss(preds, features)   # features are the ground truth in this case -> reconstruction loss

        # ---- Loop loss -----
        # ensuring edges within panel loop & return to origin

        # flatten the pattern dimention to calculate loss per panel as before
        features = features.view(-1, features.shape[-2], features.shape[-1])
        preds = preds.view(-1, preds.shape[-2], preds.shape[-1])

        panel_coords_sum = torch.zeros((features.shape[0], 2))
        panel_coords_sum = panel_coords_sum.to(device=features.device)
        self.pad_tenzor = self.pad_tenzor.to(device=features.device)
        for el_id in range(features.shape[0]):
            # iterate over elements in batch
            # loop loss per panel + we need to know each panel original length
            panel = features[el_id]
            # unpad
            bool_matrix = torch.isclose(panel, self.pad_tenzor, atol=1.e-2)
            seq_len = (~torch.all(bool_matrix, axis=1)).sum()  # only non-padded rows

            # update loss
            panel_coords_sum[el_id] = preds[el_id][:seq_len, :2].sum(axis=0)

        # panel_coords_sum = preds.sum(axis=1)[:, 0:2]  # taking only edge vectors' endpoints -- ignoring curvature coords
        panel_square_sums = panel_coords_sum ** 2  # per sum square

        # batch mean of squared norms of per-panel final points:
        loop_loss = panel_square_sums.sum() / panel_square_sums.shape[0]

        return reconstruction_loss + self.config['loop_loss_weight'] * loop_loss


# --------- PointNet++ + Patterns
class GarmentPattern3DPoint(BaseModule):
    """
        Predicting 2D pattern from 3D garment geometry
        Based on findings from GarmentPatternAE & GarmentParamsPoint 
    """
    def __init__(self, panel_elem_len, max_panel_len, max_pattern_size, data_norm={}, config={}):
        super().__init__()

        # defaults for this net
        self.config.update({
            'r1': 10, 'r2': 40,   # PointNet++
            'panel_encoding_size': 20, 
            'panel_n_layers': 3, 
            'pattern_encoding_size': 40, 
            'pattern_n_layers': 3, 
            'loop_loss_weight': 0.1, 
            'dropout': 0,
        })
        # update with input settings
        self.config.update(config) 

        # additional info
        self.config['loss'] = 'MSE with loop'
        self.config['hidden_init'] = 'kaiming_normal_'

        self.max_panel_len = max_panel_len
        self.max_pattern_size = max_pattern_size

        self.pad_tenzor = -data_norm['mean'] / data_norm['std'] if data_norm else torch.zeros(panel_elem_len)
        if not torch.is_tensor(self.pad_tenzor):
            self.pad_tenzor = torch.Tensor(self.pad_tenzor)
        self.pad_tenzor = self.pad_tenzor.repeat(max_panel_len, 1)

        # Feature extractor definition
        self.config.update({'r1': 10, 'r2': 40})  # defaults for this net
        self.config.update(config)  # from input

        self.sa1_module = SetAbstractionModule(0.5, self.config['r1'], MLP([3, 64, 64, 128]))
        self.sa2_module = SetAbstractionModule(0.25, self.config['r2'], MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSetAbstractionModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, self.config['pattern_encoding_size'])

        # Decode into pattern definition
        self.panel_decoder = LSTMDecoderModule(
            self.config['panel_encoding_size'], self.config['panel_encoding_size'], panel_elem_len, self.config['panel_n_layers'], 
            dropout=self.config['dropout']
        )
        self.pattern_decoder = LSTMDecoderModule(
            self.config['pattern_encoding_size'], self.config['pattern_encoding_size'], self.config['panel_encoding_size'], self.config['pattern_n_layers'], 
            dropout=self.config['dropout']
        )

        # init values for decoding
        self.init_submodule_params(self.panel_decoder)
        self.init_submodule_params(self.pattern_decoder)
        # leave defaults for linear layers

    def forward(self, positions_batch):
        self.device = positions_batch.device
        batch_size = positions_batch.size(0)

        # --------------- Encode ----------------
        # flatten the batch for torch-geometric batch format
        pos_flat = positions_batch.view(-1, positions_batch.size(-1))
        batch = torch.cat([
            torch.full((elem.size(0),), fill_value=i, device=self.device, dtype=torch.long) for i, elem in enumerate(positions_batch)
        ])

        # forward pass on PointNet++
        sa0_out = (None, pos_flat, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = sa3_out
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        pattern_encoding = self.lin3(x)  # YAAAAY Pattern hidden representation!!

        # ------------------- Decode ---------------
        panel_encodings_dec = self.pattern_decoder(pattern_encoding, self.max_pattern_size)

        flat_panel_encodings_dec = panel_encodings_dec.contiguous().view(-1, panel_encodings_dec.shape[-1])
        flat_panels_dec = self.panel_decoder(flat_panel_encodings_dec, self.max_panel_len)

        # back to patterns and panels
        prediction = flat_panels_dec.contiguous().view(batch_size, self.max_pattern_size, self.max_panel_len, -1)
        
        return prediction

    def init_submodule_params(self, submodule):
        """Apply custom initialization to net parameters of given submodule"""
        self.config['init'] = 'kaiming_normal_'
        for name, param in submodule.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            # leave defaults for bias

    def init_hidden(self, batch_size, n_layers, dim):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.Tensor(n_layers, batch_size, dim)
        nn.init.kaiming_normal_(hidden)
        return hidden.to(self.device)

    def loss(self, features, ground_truth):
        """Evalute loss when predicting patterns"""
        preds = self(features)

        # ---- Base reconstruction loss -----
        pattern_loss = self.regression_loss(preds, ground_truth)   # features are the ground truth in this case -> reconstruction loss

        # ---- Loop loss -----
        # ensuring edges within panel loop & return to origin

        # flatten the pattern dimention to calculate loss per panel as before
        ground_truth = ground_truth.view(-1, ground_truth.shape[-2], ground_truth.shape[-1])
        preds = preds.view(-1, preds.shape[-2], preds.shape[-1])

        panel_coords_sum = torch.zeros((ground_truth.shape[0], 2))
        panel_coords_sum = panel_coords_sum.to(device=features.device)
        self.pad_tenzor = self.pad_tenzor.to(device=features.device)
        for el_id in range(ground_truth.shape[0]):
            # iterate over elements in batch
            # loop loss per panel + we need to know each panel original length
            panel = ground_truth[el_id]
            # unpad
            bool_matrix = torch.isclose(panel, self.pad_tenzor, atol=1.e-2)
            seq_len = (~torch.all(bool_matrix, axis=1)).sum()  # only non-padded rows

            # update loss
            panel_coords_sum[el_id] = preds[el_id][:seq_len, :2].sum(axis=0)

        # panel_coords_sum = preds.sum(axis=1)[:, 0:2]  # taking only edge vectors' endpoints -- ignoring curvature coords
        panel_square_sums = panel_coords_sum ** 2  # per sum square

        # batch mean of squared norms of per-panel final points:
        loop_loss = panel_square_sums.sum() / panel_square_sums.shape[0]

        return pattern_loss + self.config['loop_loss_weight'] * loop_loss


if __name__ == "__main__":

    torch.manual_seed(125)

    a = torch.arange(1, 25, dtype=torch.float)
    dataset_gt = a.view(-1, 2, 3)
    print(dataset_gt)
    gt_batch = a.view(2, -1, 2, 3)  # ~ 2 examples in batch
    print(gt_batch)
    net = GarmentPattern3DPoint(gt_batch.shape[3], gt_batch.shape[2], gt_batch.shape[1], {'mean': dataset_gt.mean(), 'std': dataset_gt.std()})

    positions = torch.arange(1, 37, dtype=torch.float)
    features_batch = positions.view(2, -1, 3)  # note for the same batch size

    print('In batch shape: {}; Out batch shape: {}'.format(features_batch.shape, gt_batch.shape))
    print(net(features_batch)) 
    loss = net.loss(features_batch, gt_batch)
    print(loss)
    loss.backward()  # check it doesn't fail