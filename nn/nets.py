import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geometric

class ShirtfeaturesMLP(nn.Module):
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


class GarmentParamsMLP(nn.Module):
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
                            max_num_neighbors=64)
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


class GarmentParamsPoint(torch.nn.Module):
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


if __name__ == "__main__":
    net = GarmentParamsPoint(10)
    # print(net)

    a = torch.arange(1, 31, dtype=torch.float)
    batch = a.view(2, -1, 3)  # ~ 2 examples, 5 3D points each

    print(net(batch))  # should have 2 x 10 shape -- per example prediction