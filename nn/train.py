
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import wandb as wb

import dataloaders as dl
import trainer
import nets


# --------- Reproducibility
# see https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)

# when using cuda
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
data_location = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'

# Basic Parameters
batch_size = 64
epochs_num = 100
learning_rate = 0.001
logdir = './logdir'

#-------- DATA --------
# Initial load
shirt_dataset = dl.ParametrizedShirtDataSet(Path(data_location), 
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

training_loader = DataLoader(training_set, batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size)


# model
model = nets.ShirtfeaturesMLP()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

# loss function
regression_loss = nn.MSELoss()

# init Weights&biases run
#os.environ['WANDB_MODE'] = 'dryrun'

wb.init(name="refactoring-no-norm", project='Test-Garments-Reconstruction')

wb.watch(model, log='all')

trainer.fit(model, regression_loss, optimizer, scheduler, training_loader, validation_loader)

print ("Finished training")

# loss on validation set
model.eval()
with torch.no_grad():
    valid_loss = sum([regression_loss(model(batch['features']), batch['pattern_params']) for batch in validation_loader]) 

print ('Validation loss: {}'.format(valid_loss))


# save prediction for validation to file
model.eval()
with torch.no_grad():
    batch = next(iter(validation_loader))    # might have some issues, see https://github.com/pytorch/pytorch/issues/1917
    shirt_dataset.save_prediction_batch(model(batch['features']), batch['name'])