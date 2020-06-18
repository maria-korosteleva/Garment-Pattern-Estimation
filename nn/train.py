
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# My modules
import customconfig
import data
from trainer import Trainer
from tester import Tester
import nets

# init
datapath = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
system_info = customconfig.Properties('./system.json')
trainer = Trainer(
    system_info['wandb_username'],
    project_name='Test-Garments-Reconstruction', 
    run_name='with_tester', 
    resume_run_id=None) 

# Data load and split
shirts = trainer.use_dataset(data.ParametrizedShirtDataSet(Path(datapath)), valid_percent=10)

# model
trainer.init_randomizer()
model = nets.ShirtfeaturesMLP()

# fit
trainer.fit(model)

# --------------- Final tests on validation set --------------
tester = Tester(model)

valid_loss = tester.metrics(shirts, 'validation')
print ('Validation loss: {}'.format(valid_loss))

# save prediction for validation to file
model.eval()
with torch.no_grad():
    batch = next(iter(shirts.loader_validation))    # might have some issues, see https://github.com/pytorch/pytorch/issues/1917
    shirts.dataset.save_prediction_batch(model(batch['features']), batch['name'])