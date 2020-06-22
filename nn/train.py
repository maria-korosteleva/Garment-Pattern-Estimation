
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# My modules
import customconfig
import data
from trainer import Trainer
from prediction import PredictionManager
import nets

# init
datapath = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
system_info = customconfig.Properties('./system.json')
trainer = Trainer(
    system_info['wandb_username'],
    project_name='Test-Garments-Reconstruction', 
    run_name='predicting', 
    resume_run_id=None) 

# Data load and split
shirts = trainer.use_dataset(data.ParametrizedShirtDataSet(Path(datapath)), valid_percent=10)
# model
trainer.init_randomizer()
model = nets.ShirtfeaturesMLP()
# fit
trainer.fit(model)

# --------------- Final tests on validation set --------------
tester = PredictionManager(model)
valid_loss = tester.metrics(shirts, 'validation')
print ('Validation loss: {}'.format(valid_loss))

# save prediction for validation to file
tester.predict(shirts, 'validation')