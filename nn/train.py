
from pathlib import Path


import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import wandb as wb

# My modules
import dataloaders as dl
from trainer import Trainer
import nets
from customconfig import Properties

# Basic Parameters
trainer = Trainer(
    r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest', 
    project_name='Test-Garments-Reconstruction', 
    run_name='refactoring-config')

trainer.init_randomizer()

#-------- DATA --------
# Initial load
shirt_dataset = dl.ParametrizedShirtDataSet(Path(wb.config.dataset), 
                                  dl.SampleToTensor())
# Data normalization
# mean, std = dl.get_mean_std(DataLoader(shirt_dataset, 100))
# shirt_dataset = dl.ParametrizedShirtDataSet(Path(data_location), transforms.Compose([dl.SampleToTensor(), dl.NormalizeInputfeatures(mean, std)]))

# Data load and split
valid_size = (int) (len(shirt_dataset) / 10)
# split is RANDOM. Might affect performance
training_set, validation_set = torch.utils.data.random_split(
    shirt_dataset, 
    (len(shirt_dataset) - valid_size, valid_size))

print ('Split: {} / {}'.format(len(training_set), len(validation_set)))

training_loader = DataLoader(training_set, trainer.setup['batch_size'], shuffle=True)
validation_loader = DataLoader(validation_set, trainer.setup['batch_size'])


# model
model = nets.ShirtfeaturesMLP()

# ----- Fit ---------

trainer.fit(model, training_loader, validation_loader)

# --------------- loss on validation set --------------
model.eval()
with torch.no_grad():
    valid_loss = sum([trainer.regression_loss(model(batch['features']), batch['pattern_params']) for batch in validation_loader]) 

print ('Validation loss: {}'.format(valid_loss))


# save prediction for validation to file
model.eval()
with torch.no_grad():
    batch = next(iter(validation_loader))    # might have some issues, see https://github.com/pytorch/pytorch/issues/1917
    shirt_dataset.save_prediction_batch(model(batch['features']), batch['name'])