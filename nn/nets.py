import torch.nn as nn
import torch.functional as F


class ShirtfeaturesMLP(nn.Module):
    """MLP for training on shirts dataset. Assumes 100 features parameters used"""
    
    def __init__(self):
        super().__init__()
        
        # layers definitions
        self.sequence = nn.Sequential(
            nn.Linear(1500, 300),  # nn.Linear(36756, 3000),
            nn.ReLU(), 
            nn.Linear(300, 300),  # nn.Linear(3000, 300)
            nn.ReLU(), 
            nn.Linear(300, 60),
            nn.ReLU(),
            nn.Linear(60, 9)
        )
    
    def forward(self, x_batch):
        #print (x_batch)
        
        return self.sequence(x_batch)

if __name__ == "__main__":
    net = ShirtfeaturesMLP()

    print(net)