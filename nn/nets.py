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
    def __init__(self, out_size, config={'r1': 0.2, 'r2': 0.4}):
        super().__init__()

        self.config = config

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
    def __init__(self, in_elem_len, max_seq_len, config={}):
        super().__init__()

        # defaults
        self.config = {'hidden_dim_enc': 20, 'hidden_dim_dec': 20, 'n_layers': 3, 'loop_loss_weight': 1}
        self.config.update(config)

        # loss info
        self.config['loss'] = 'MSE Reconstruction with loop'

        self.max_seq_len = max_seq_len

        # encode
        self.seq_encoder = nn.LSTM(
            in_elem_len, 
            self.config['hidden_dim_enc'], self.config['n_layers'], 
            batch_first=True)

        # decode
        self.seq_decoder = nn.LSTM(
            self.config['hidden_dim_enc'], 
            self.config['hidden_dim_dec'], self.config['n_layers'], 
            batch_first=True)

        # post-process
        self.lin = nn.Linear(self.config['hidden_dim_dec'], in_elem_len)

    def forward(self, x):
        self.device = x.device
        batch_size = x.size(0)
        
        # --- encode --- 
        hidden_init, cell_init = self.init_hidden(batch_size), self.init_hidden(batch_size)
        out, (hidden, _) = self.seq_encoder(x, (hidden_init, cell_init))
        # final encoding is the last output == hidden of last layer 
        encoding = hidden[-1]

        # --- decode ---
        # propagate encoding for needed seq_len
        dec_input = encoding.unsqueeze(1).repeat(1, self.max_seq_len, 1)  # along sequence dimention
        # init memory with zeros (not with encoder state) for future indeendent use of decoder 
        dec_hidden_init, dec_cell_init = self.init_hidden(batch_size), self.init_hidden(batch_size)
        out, hidden = self.seq_decoder(dec_input, (dec_hidden_init, dec_cell_init))
        
        # --- back to original format --- 
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.config['hidden_dim_dec'])
        out = self.lin(out)
        # back to sequence
        out = out.contiguous().view(batch_size, self.max_seq_len, -1)
        
        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.config['n_layers'], batch_size, self.config['hidden_dim_enc'])
        return hidden.to(self.device)

    def loss(self, features, ground_truth):
        """Override base class loss calculation to use reconstruction loss"""
        preds = self(features)

        # Base reconstruction loss
        reconstruction_loss = self.regression_loss(preds, features)   # features are the ground truth in this case -> reconstruction loss

        # ensuring edges within panel loop & return to origin
        panel_coords_sum = preds.sum(axis=1)[:, 0:2]  # taking only edge vectors' endpoints -- ignoring curvature coords
        panel_square_sums = panel_coords_sum ** 2  # per sum square

        # batch mean of squared norms of per-panel final points:
        loop_loss = panel_square_sums.sum() / panel_square_sums.shape[0]

        return reconstruction_loss + self.config['loop_loss_weight'] * loop_loss


if __name__ == "__main__":

    torch.manual_seed(125)

    a = torch.arange(1, 25, dtype=torch.float)
    batch = a.view(2, -1, 3)  # ~ 2 examples in batch

    net = GarmentPanelsAE(batch.shape[2], batch.shape[1])

    print('In batch shape: {}'.format(batch.shape))
    print(net(batch))  # should have 2 x 10 shape -- per example prediction
    loss = net.loss(batch, None)
    print(loss)
    loss.backward()  # check it doesn't fail