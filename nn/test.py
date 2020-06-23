"""Evaluate a model on the data"""

from pathlib import Path

# My modules
import customconfig
import data
from trainer import Trainer
from experiment import WandbRunWrappper
from prediction import PredictionManager
import nets

# init
datapath = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
system_info = customconfig.Properties('./system.json')
experiment = WandbRunWrappper(
    system_info['wandb_username'],
    project_name='Test-Garments-Reconstruction', 
    run_name='wb_wrapper', 
    run_id='3kzta02g')  # finished experiment

# Load Model
model = nets.ShirtfeaturesMLP()
model.load_state_dict(experiment.load_final_model(to_path=Path('./wandb')))
# model.load_state_dict(experiment.load_checkpoint_file(3, to_path=Path('./wandb'))['model_state_dict'])

# Load data for eval
split, batch_size = experiment.data_info()  # note that run is not initialized -- we use info from finished run
shirts_wrapper = data.DatasetWrapper(
    data.ParametrizedShirtDataSet(datapath), 
    known_split=split, batch_size=batch_size)

# Evaluate
tester = PredictionManager()
valid_loss = tester.metrics(model, shirts_wrapper, 'validation')
print ('Validation loss: {}'.format(valid_loss))

experiment.add_statistic('valid_metrics', valid_loss)

# save prediction for validation to file
tester.predict(model, shirts_wrapper, 'validation')