
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# My modules
import customconfig
import data
from trainer import Trainer
import nets

# init
datapath = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
system_info = customconfig.Properties('./system.json')
trainer = Trainer(
    system_info['wandb_username'],
    project_name='Test-Garments-Reconstruction', 
    run_name='resume', 
    resume_run_id='2gxm4sfg') 

# Data load and split
shirts = trainer.use_dataset(data.ParametrizedShirtDataSet(Path(datapath)), valid_percent=10)

# model
trainer.init_randomizer()
model = nets.ShirtfeaturesMLP()

# fit
trainer.fit(model)

# --------------- loss on validation set --------------
model.eval()
with torch.no_grad():
    valid_loss = sum([trainer.regression_loss(model(batch['features']), batch['pattern_params']) for batch in shirts.loader_validation]) 

print ('Validation loss: {}'.format(valid_loss))

# save prediction for validation to file
model.eval()
with torch.no_grad():
    batch = next(iter(shirts.loader_validation))    # might have some issues, see https://github.com/pytorch/pytorch/issues/1917
    shirts.dataset.save_prediction_batch(model(batch['features']), batch['name'])