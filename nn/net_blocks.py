"""Basic building blocks for custom neural network architectures"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geometric


# --------- PointNet++ modules -------
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py

class _SetAbstractionModule(nn.Module):
    """Performes PointNet feature extraction in local areas (around sampled centroids) of given sets"""
    def __init__(self, ratio, conv_radius, per_point_nn):
        super().__init__()
        self.ratio = ratio  # for controlling number of centroids
        self.radius = conv_radius
        self.conv = geometric.PointConv(per_point_nn)

    def forward(self, features, pos, batch):
        idx = geometric.fps(pos, batch, ratio=self.ratio)
        row, col = geometric.radius(pos, pos[idx], self.radius, batch, batch[idx],
                            max_num_neighbors=25)
        edge_index = torch.stack([col, row], dim=0)
        features = self.conv(features, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return features, pos, batch

class _GlobalSetAbstractionModule(nn.Module):
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

def _MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])

class PointNetPlusPlus(nn.Module):
    """
        Module for extracting latent representation of 3D geometry.
        Based on PointNet++
        NOTE architecture is agnostic of number of input points
    """
    def __init__(self, out_size, config={}):
        super().__init__()

        self.config = {'r1': 3, 'r2': 4, 'r3': 5, 'r4': 7}  # defaults for this net
        self.config.update(config)  # from input

        self.sa1_module = _SetAbstractionModule(0.2, self.config['r1'], _MLP([3, 64, 64, 128]))
        self.sa2_module = _SetAbstractionModule(0.25, self.config['r2'], _MLP([128 + 3, 128, 128, 128]))
        self.sa3_module = _SetAbstractionModule(0.25, self.config['r3'], _MLP([128 + 3, 128, 128, 128]))
        self.sa4_module = _SetAbstractionModule(0.25, self.config['r4'], _MLP([128 + 3, 128, 128, 256]))
        self.sa_last_module = _GlobalSetAbstractionModule(_MLP([256 + 3, 256, 512, 1024]))

        self.lin = nn.Linear(1024, out_size)

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
        sa4_out = self.sa4_module(*sa3_out)
        sa_last_out = self.sa_last_module(*sa4_out)
        out, _, _ = sa_last_out
        out = self.lin(out)
        return out

# ------------- EdgeConv ----------
# https://github.com/AnTao97/dgcnn.pytorch/blob/master/model.py

class EdgeConvFeatures(nn.Module):
    """Extracting feature vector from 3D point cloud based on Edge convolutions from Paper “Dynamic Graph CNN for Learning on Point Clouds”"""
    def __init__(self, out_size, config={}):
        super().__init__()

        self.config = {'conv_depth': 3}  # defaults for this net
        self.config.update(config)  # from input

        # DynamicEdgeConv!!!!!!
        self.conv1 = geometric.DynamicEdgeConv(_MLP([2 * 3, 64, 64, 32]), k=10, aggr='max')
        self.conv2 = geometric.DynamicEdgeConv(_MLP([2 * 32, 64, 64, 32]), k=10, aggr='max')
        self.conv3 = geometric.DynamicEdgeConv(_MLP([2 * 32, 64, 64, 32]), k=10, aggr='max')

        self.lin = nn.Linear(32, out_size)

    def forward(self, positions):
        batch_size = positions.size(0)
        n_vertices = positions.size(1)
        # flatten the batch for torch-geometric batch format
        pos_flat = positions.view(-1, positions.size(-1))
        batch = torch.cat([
            torch.full((elem.size(0),), fill_value=i, device=positions.device, dtype=torch.long) for i, elem in enumerate(positions)
        ])

        # Vertex features
        out = self.conv1(pos_flat, batch)
        out = self.conv2(out, batch)
        out = self.conv3(out, batch)  # n_points x length_features
        # reshape back into batch 
        out = out.contiguous().view(batch_size, n_vertices, -1)

        # aggregate features from vertices
        out = out.max(dim=-2, keepdim=False)[0]

        # post-processing
        out = self.lin(out)

        return out

# ------------- Sequence modules -----------
def _init_tenzor(*shape, device='cpu', init_type=''):
    """shortcut to create & initialize tenzors on a given device.  """
    # TODO suport other init types 
    if not init_type: # zeros by default
        new_tenzor = torch.zeros(shape)
    elif 'kaiming_normal' in init_type:
        new_tenzor = torch.empty(shape)
        nn.init.kaiming_normal_(new_tenzor)
    else:  
        raise NotImplementedError('{} tenzor initialization is not implemented'.format(init_type))

    return new_tenzor.to(device)

def _init_weights(module, init_type=''):
    """Initialize weights of provided module with requested init type"""
    if not init_type:
        # do not re-initialize, leave default pytorch init
        return
    for name, param in module.named_parameters():
        if 'weight' in name:
            if 'kaiming_normal' in init_type:
                nn.init.kaiming_normal_(param)
            else:
                raise NotImplementedError('{} weight initialization is not implemented'.format(init_type))
    # leave defaults for bias


class LSTMEncoderModule(nn.Module):
    """A wrapper for LSTM targeting encoding task"""
    def __init__(self, elem_len, encoding_size, n_layers, dropout=0, custom_init='kaiming_normal'):
        super().__init__()
        self.custom_init = custom_init
        self.n_layers = n_layers
        self.encoding_size = encoding_size

        self.lstm = nn.LSTM(
            elem_len, encoding_size, n_layers, 
            dropout=dropout, batch_first=True)

        _init_weights(self.lstm, init_type=custom_init)

    def forward(self, batch_sequence):
        device = batch_sequence.device
        batch_size = batch_sequence.size(0)
        
        # --- encode --- 
        hidden_init = _init_tenzor(self.n_layers, batch_size, self.encoding_size, device=device, init_type=self.custom_init)
        cell_init = _init_tenzor(self.n_layers, batch_size, self.encoding_size, device=device, init_type=self.custom_init)
        _, (hidden, _) = self.lstm(batch_sequence, (hidden_init, cell_init))

        # final encoding is the last output == hidden of last layer 
        return hidden[-1]


class LSTMDecoderModule(nn.Module):
    """A wrapper for LSTM targeting decoding task"""
    def __init__(self, encoding_size, hidden_size, out_elem_size, n_layers, dropout=0, custom_init='kaiming_normal'):
        super().__init__()
        self.custom_init = custom_init
        self.n_layers = n_layers
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.out_elem_size = out_elem_size

        self.lstm = nn.LSTM(encoding_size, hidden_size, n_layers, 
                            dropout=dropout, batch_first=True)

        # post-process to match the desired outut shape
        self.lin = nn.Linear(hidden_size, out_elem_size)

        # initialize
        _init_weights(self.lstm, init_type=custom_init)

    def forward(self, batch_enc, out_len):
        """out_len specifies the length of the output sequence to produce"""
        device = batch_enc.device
        batch_size = batch_enc.size(0)
        
        # propagate encoding for needed seq_len
        dec_input = batch_enc.unsqueeze(1).repeat(1, out_len, 1)  # along sequence dimention

        # decode
        hidden_init = _init_tenzor(self.n_layers, batch_size, self.hidden_size, device=device, init_type=self.custom_init)
        cell_init = _init_tenzor(self.n_layers, batch_size, self.hidden_size, device=device, init_type=self.custom_init)
        out, _ = self.lstm(dec_input, (hidden_init, cell_init))
        
        # back to requested format
        # reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.lin(out)
        # back to sequence
        out = out.contiguous().view(batch_size, out_len, -1)

        return out


class LSTMDoubleReverseDecoderModule(nn.Module):
    """A wrapper for LSTM targeting decoding task that decodes the sequence in the reverse order, 
    and then processes it in the forward order to refine the reconstuction"""
    def __init__(self, encoding_size, hidden_size, out_elem_size, n_layers, dropout=0, custom_init='kaiming_normal'):
        super().__init__()
        self.custom_init = custom_init
        self.n_layers = n_layers
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.out_elem_size = out_elem_size

        # revrese & forward models share the architecture but not the weights
        self.lstm_reverse = nn.LSTM(encoding_size, hidden_size, n_layers, 
                                    dropout=dropout, batch_first=True)
        self.lstm_forward = nn.LSTM(hidden_size + encoding_size, hidden_size, n_layers, 
                                    dropout=dropout, batch_first=True)

        # post-process to match the desired outut shape
        self.lin = nn.Linear(hidden_size, out_elem_size)

        # initialize
        _init_weights(self.lstm_reverse, init_type=custom_init)
        _init_weights(self.lstm_forward, init_type=custom_init)

    def forward(self, batch_enc, out_len):
        """out_len specifies the length of the output sequence to produce"""
        device = batch_enc.device
        batch_size = batch_enc.size(0)
        
        # propagate encoding for needed seq_len
        dec_input = batch_enc.unsqueeze(1).repeat(1, out_len, 1)  # along sequence dimention

        # decode reversed sequence
        hidden_init = _init_tenzor(self.n_layers, batch_size, self.hidden_size, device=device, init_type=self.custom_init)
        cell_init = _init_tenzor(self.n_layers, batch_size, self.hidden_size, device=device, init_type=self.custom_init)
        out, state = self.lstm_reverse(dec_input, (hidden_init, cell_init))
        
        # decode forward sequence
        out = torch.flip(out, [1])
        out = torch.cat([out, dec_input], -1)  # skip connection with original input
        out, _ = self.lstm_forward(out, state)  # pass the state from previous module for additional info

        # back to requested format
        # reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.lin(out)
        # back to sequence
        out = out.contiguous().view(batch_size, out_len, -1)

        return out




# Quick tests
if __name__ == "__main__":

    torch.manual_seed(125)

    positions = torch.arange(1, 7, dtype=torch.float)  # 37
    # features_batch = positions.view(2, -1, 3)  # note for the same batch size
    features_batch = positions.view(-1, 3) 

    print('In batch shape: {}'.format(features_batch.shape))

    # net = EdgeConvFeatures(5)
    net = LSTMDoubleReverseDecoderModule(3, 4, 4, 2)

    print(net(features_batch, 4))
