"""Evaluate a model on the data"""

from pathlib import Path
import torch

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys,inspect
currentdir = os.path.dirname(os.path.realpath(__file__) )
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

# My modules
import customconfig
import data
import metrics
import nets
from trainer import Trainer
from experiment import WandbRunWrappper

# --------------- from experimnet ---------
system_info = customconfig.Properties('./system.json')
experiment = WandbRunWrappper(
    system_info['wandb_username'],
    project_name='Garments-Reconstruction', 
    run_name='pants-panel-disentanglement', 
    run_id='3txt9wae')  # finished experiment

if not experiment.is_finished():
    print('Warning::Evaluating unfinished experiment')

# -------- data -------
# data_config also contains the names of datasets to use
split, batch_size, data_config = experiment.data_info()  # note that run is not initialized -- we use info from finished run

data_config.update({'obj_filetag': 'sim'})  # scan imitation stats

dataset = data.Garment3DPatternFullDataset(
    system_info['datasets_path'], data_config, gt_caching=True, feature_caching=True)

print(dataset.config)
print('Batch: {}, Split: {}'.format(batch_size, split))

datawrapper = data.DatasetWrapper(dataset, known_split=split, batch_size=batch_size)

# ----- Model architecture -----
model = nets.GarmentFullPattern3DDisentangle(dataset.config, experiment.NN_config())

model.load_state_dict(experiment.load_best_model(device='cuda:0')['model_state_dict'])

# ------- Evaluate --------
valid_loss = metrics.eval_metrics(model, datawrapper, 'validation')
print('Validation metrics: {}'.format(valid_loss))
# valid_breakdown = metrics.eval_metrics(model, datawrapper, 'valid_per_data_folder')
# print('Validation metrics per dataset: {}'.format(valid_breakdown))

test_metrics = metrics.eval_metrics(model, datawrapper, 'test')
print('Test metrics: {}'.format(test_metrics))
# test_breakdown = metrics.eval_metrics(model, datawrapper, 'test_per_data_folder')
# print('Test metrics per dataset: {}'.format(test_breakdown))

# print(dataset[276]['features'])  # first element of validation set

experiment.add_statistic('valid_on_best', valid_loss)
# experiment.add_statistic('valid', valid_breakdown)
experiment.add_statistic('test_on_best', test_metrics)
# experiment.add_statistic('test', test_breakdown)

# -------- Predict ---------
# save prediction for validation to file
# prediction_path = datawrapper.predict(model, save_to=Path(system_info['output']), sections=['validation', 'test'])
# print('Saved to {}'.format(prediction_path))
# # # reflect predictions info in expetiment
# experiment.add_statistic('pred_folder', prediction_path.name)

# art_name = 'multi-data' if len(datawrapper.dataset.data_folders) > 1 else datawrapper.dataset.data_folders[0]  # + '-scan'
# experiment.add_artifact(prediction_path, art_name, 'result')