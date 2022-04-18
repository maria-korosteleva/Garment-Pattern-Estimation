"""Evaluate a model on the data"""

from pathlib import Path
import torch
import torch.nn as nn

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys,inspect
currentdir = os.path.dirname(os.path.realpath(__file__) )
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

# My modules
import customconfig
import data
from metrics.eval_utils import eval_metrics
import nets
from trainer import Trainer
from experiment import WandbRunWrappper

# --------------- from experimnet ---------
system_info = customconfig.Properties('./system.json')
experiment = WandbRunWrappper(
    system_info['wandb_username'],
    project_name='Garments-Reconstruction', 
    run_name='NeuralTailor-Train',
    run_id='uazmw1ro')  # finished experiment
    # run_name='No-Loop-Filt-Att-Condenced',
    # run_id='2pbrilln')  # finished experiment

if not experiment.is_finished():
    print('Warning::Evaluating unfinished experiment')

# -------- data -------
dataset_list = [
    'jacket_hood_sleeveless_150',
    'skirt_waistband_150', 
    'tee_hood_150',
    'jacket_sleeveless_150',
    'dress_150',
    'jumpsuit_150',
    'wb_jumpsuit_sleeveless_150'
]

# data_config also contains the names of datasets to use
split, batch_size, data_config = experiment.data_info()  # note that run is not initialized -- we use info from finished run

data_config.update({'obj_filetag': 'sim'})  #  DEBUG sim\scan imitation stats
data_config.update(data_folders=dataset_list)
# data_config.pop('max_num_stitches', None)  # NOTE forces re-evaluation of max pattern sizes (but not standardization stats) 
data_config.update(max_datapoints_per_type=150)

if not experiment.is_finished():
    # NOTE Use files appropriate for the experiment
    data_config.update({'panel_classification': './nn/data_configs/panel_classes_condenced.json'})   
    data_config.update({'filter_by_params': './nn/data_configs/param_filter.json'})

batch_size = 5   # fit on less powerfull machines

print(data_config['class'])

if 'class' in data_config:
    data_class = getattr(data, data_config['class'])
    dataset = data_class(system_info['datasets_path'] + '/test', data_config, gt_caching=True, feature_caching=True)
else:
    dataset = data.GarmentStitchPairsDataset(
        system_info['datasets_path'] + '/test', data_config, gt_caching=True, feature_caching=True)

datawrapper = data.DatasetWrapper(dataset, batch_size=batch_size)  # NOTE no split given -- evaluating on the full loaded dataset!!

# ----- Model architecture -----
model_class = getattr(nets, experiment.NN_config()['model'])
model = model_class(dataset.config, experiment.NN_config(), experiment.NN_config()['loss'])

if 'device_ids' in experiment.NN_config():  # model from multi-gpu training case
    model = nn.DataParallel(model, device_ids=['cuda:0'])
model.load_state_dict(experiment.load_best_model(device='cuda:0')['model_state_dict'])

# ------- Evaluate --------
loss = eval_metrics(model, datawrapper, 'full')
print('Full metrics on unseen set: {}'.format(loss))
breakdown = eval_metrics(model, datawrapper, 'full_per_data_folder')
print('Metrics per dataset: {}'.format(breakdown))

# # -------- Predict ---------
prediction_path = datawrapper.predict(model, save_to=Path(system_info['output']), sections=['full'])
print('Saved to {}'.format(prediction_path))

if experiment.is_finished():  # records won't be updates for unfinished experiment anyway
    # ---------- Log to the experiment -----------
    experiment.add_statistic('unseen_full', loss)
    experiment.add_statistic('unseen', breakdown)
    experiment.add_statistic('unseen_folders', dataset_list)

    # reflect predictions info in expetiment
    experiment.add_statistic('unseen_scan_pred_folder', prediction_path.name)

    art_name = 'multi-data-unseen' if len(datawrapper.dataset.data_folders) > 1 else datawrapper.dataset.data_folders[0] + '-unseen'
    experiment.add_artifact(prediction_path, art_name, 'result')
