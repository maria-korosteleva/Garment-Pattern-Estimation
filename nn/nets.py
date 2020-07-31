import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geometric

class BaseModule(nn.Module):
    """Base interface for neural nets"""
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

        self.max_seq_len = max_seq_len

        self.pad_tenzor = -data_norm['mean'] / data_norm['std'] if data_norm else torch.zeros(in_elem_len)
        if not torch.is_tensor(self.pad_tenzor):
            self.pad_tenzor = torch.Tensor(self.pad_tenzor)
        self.pad_tenzor = self.pad_tenzor.repeat(max_seq_len, 1)

        # encode
        self.seq_encoder = nn.LSTM(
            in_elem_len, 
            self.config['hidden_dim_enc'], self.config['n_layers'], 
            dropout=self.config['dropout'],
            batch_first=True)

        # decode
        self.seq_decoder = nn.LSTM(
            self.config['hidden_dim_enc'], 
            self.config['hidden_dim_dec'], self.config['n_layers'], 
            dropout=self.config['dropout'],
            batch_first=True)

        # post-process
        self.lin = nn.Linear(self.config['hidden_dim_dec'], in_elem_len)

        # init values
        self.init_net_params()

    def forward(self, x):
        self.device = x.device
        batch_size = x.size(0)
        
        # --- encode --- 
        hidden_init = self.init_hidden(batch_size, self.config['n_layers'], self.config['hidden_dim_enc'])
        cell_init =  self.init_hidden(batch_size, self.config['n_layers'], self.config['hidden_dim_enc'])
        out, (hidden, _) = self.seq_encoder(x, (hidden_init, cell_init))
        # final encoding is the last output == hidden of last layer 
        encoding = hidden[-1]

        # --- decode ---
        # propagate encoding for needed seq_len
        dec_input = encoding.unsqueeze(1).repeat(1, self.max_seq_len, 1)  # along sequence dimention
        # init memory with zeros (not with encoder state) for future indeendent use of decoder 
        dec_hidden_init = self.init_hidden(batch_size, self.config['n_layers'], self.config['hidden_dim_dec'])
        dec_cell_init =  self.init_hidden(batch_size, self.config['n_layers'], self.config['hidden_dim_dec'])
        out, hidden = self.seq_decoder(dec_input, (dec_hidden_init, dec_cell_init))
        
        # --- back to original format --- 
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.config['hidden_dim_dec'])
        out = self.lin(out)
        # back to sequence
        out = out.contiguous().view(batch_size, self.max_seq_len, -1)
        
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

        # leave defaults for linear layer

    def init_hidden(self, batch_size, n_layers, dim):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.Tensor(n_layers, batch_size, dim)
        nn.init.kaiming_normal_(hidden)
        return hidden.to(self.device)

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
        Based on finding from GarmentPanelsAE
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

        self.max_panel_len = max_panel_len

        self.pad_tenzor = -data_norm['mean'] / data_norm['std'] if data_norm else torch.zeros(in_elem_len)
        if not torch.is_tensor(self.pad_tenzor):
            self.pad_tenzor = torch.Tensor(self.pad_tenzor)
        self.pad_tenzor = self.pad_tenzor.repeat(max_panel_len, 1)

        # --- panel-level ---- 
        # encode
        self.panel_encoder = nn.LSTM(
            in_elem_len, 
            self.config['panel_encoding_size'], self.config['panel_n_layers'], 
            dropout=self.config['dropout'],
            batch_first=True)

        # decode
        self.panel_decoder = nn.LSTM(
            self.config['panel_encoding_size'], 
            self.config['panel_encoding_size'], self.config['panel_n_layers'], 
            dropout=self.config['dropout'],
            batch_first=True)

        # post-process on panel-level
        self.panel_lin = nn.Linear(self.config['panel_encoding_size'], in_elem_len)

        # ----- patten level ------
        # given the panel encodings, combine them into pattern encoding
        # encode
        self.pattern_encoder = nn.LSTM(
            self.config['panel_encoding_size'], 
            self.config['pattern_encoding_size'], self.config['pattern_n_layers'],
            dropout=self.config['dropout'],
            batch_first=True
        )

        self.pattern_decoder = nn.LSTM(
            self.config['pattern_encoding_size'], 
            self.config['pattern_encoding_size'], self.config['pattern_n_layers'],
            dropout=self.config['dropout'],
            batch_first=True
        )

        self.pattern_lin = nn.Linear(self.config['pattern_encoding_size'], self.config['panel_encoding_size'])

        # init values
        self.init_submodule_params(self.panel_encoder)
        self.init_submodule_params(self.panel_decoder)
        self.init_submodule_params(self.pattern_encoder)
        self.init_submodule_params(self.pattern_decoder)
        # leave defaults for linear layers

    def forward(self, patterns_batch):
        self.device = patterns_batch.device
        batch_size = patterns_batch.size(0)
        pattern_size = patterns_batch.size(1)
        num_panels = batch_size * pattern_size

        # --------------- Encode ----------------
        # ----- Panel-level -----
        # flatten -- view simply as a list of panels to apply encoding per panel
        all_panels = patterns_batch.contiguous().view(num_panels, patterns_batch.shape[-2], patterns_batch.shape[-1])

        hidden_init = self.init_hidden(num_panels, self.config['panel_n_layers'], self.config['panel_encoding_size'])
        cell_init =  self.init_hidden(num_panels, self.config['panel_n_layers'], self.config['panel_encoding_size'])
        out, (hidden, _) = self.panel_encoder(all_panels, (hidden_init, cell_init))
        # final encoding is the last output == hidden of last layer 
        panel_encodings = hidden[-1]
        
        # ---- Pattern-level -----
        panel_encodings = panel_encodings.contiguous().view(batch_size, pattern_size, -1)  # get back to patterns
        hidden_init = self.init_hidden(batch_size, self.config['pattern_n_layers'], self.config['pattern_encoding_size'])
        cell_init =  self.init_hidden(batch_size, self.config['pattern_n_layers'], self.config['pattern_encoding_size'])
        out, (hidden, _) = self.pattern_encoder(panel_encodings, (hidden_init, cell_init))

        pattern_encoding = hidden[-1]   # YAAAAY Pattern hidden representation!!

        # ------------------- Decode ---------------
        # ---- Pattern-level -----
        pattern_dec_input = pattern_encoding.unsqueeze(1).repeat(1, pattern_size, 1)  # along sequence dimention
        hidden_init = self.init_hidden(batch_size, self.config['pattern_n_layers'], self.config['pattern_encoding_size'])
        cell_init =  self.init_hidden(batch_size, self.config['pattern_n_layers'], self.config['pattern_encoding_size'])
        out, hidden = self.pattern_decoder(pattern_dec_input, (hidden_init, cell_init))
        # Flatten: view as just a list of panel encodings to apply lin layer per panel
        out = out.contiguous().view(-1, self.config['pattern_encoding_size'])
        all_panels_encodings = self.pattern_lin(out)  # now we have coorect size for predicted panel encodings

        # ----- Panel-level -----
        dec_input = all_panels_encodings.unsqueeze(1).repeat(1, self.max_panel_len, 1)  # along sequence dimention
        hidden_init = self.init_hidden(num_panels, self.config['panel_n_layers'], self.config['panel_encoding_size'])
        cell_init =  self.init_hidden(num_panels, self.config['panel_n_layers'], self.config['panel_encoding_size'])
        out, hidden = self.panel_decoder(dec_input, (hidden_init, cell_init))
        # Falatten: as simple list of edges without panel\pattern grouping to apply lin layer per edge
        out = out.contiguous().view(-1, self.config['panel_encoding_size'])
        out = self.panel_lin(out)  # correct size for predicted edges
        # back to patterns and panels
        prediction = out.contiguous().view(batch_size, pattern_size, self.max_panel_len, -1)
        
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


if __name__ == "__main__":

    torch.manual_seed(125)

    a = torch.arange(1, 25, dtype=torch.float)
    dataset = a.view(-1, 2, 3)
    print(dataset)
    batch = a.view(2, -1, 2, 3)  # ~ 2 examples in batch
    print(batch)
    net = GarmentPatternAE(batch.shape[3], batch.shape[2], {'mean': dataset.mean(), 'std': dataset.std()})

    print('In batch shape: {}'.format(batch.shape))
    print(net(batch)) 
    loss = net.loss(batch, None)
    print(loss)
    loss.backward()  # check it doesn't fail