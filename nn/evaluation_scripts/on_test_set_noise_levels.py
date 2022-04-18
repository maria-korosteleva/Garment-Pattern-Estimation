"""Evaluate a model on the data"""

from pathlib import Path
import torch
from torch import nn
import argparse
import json
from datetime import datetime

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys,inspect
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# My modules
import customconfig
import data
from metrics.eval_utils import eval_metrics
import nets
from trainer import Trainer
from experiment import ExperimentWrappper


# --------------- from experimnet ---------
system_info = customconfig.Properties('./system.json')
experiment = ExperimentWrappper(
    system_info['wandb_username'],
    project_name='Garments-Reconstruction', 
    run_name='Filtered-att-data-condenced-classes',
    run_id='390wuxbm')  # finished experiment

if not experiment.is_finished():
    print('Warning::Evaluating unfinished experiment')

noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]  

# -------- data -------
# data_config also contains the names of datasets to use
split, batch_size, data_config = experiment.data_info()  # note that run is not initialized -- we use info from finished run
data_config.update({'obj_filetag': 'sim'})  # scan imitation stats

# ----- Model architecture -----
loss_config = experiment.NN_config()['loss']
model_class = getattr(nets, experiment.NN_config()['model'])
model = model_class(data_config, experiment.NN_config(), loss_config)
if 'device_ids' in experiment.NN_config():  # model from multi-gpu training case
    model = nn.DataParallel(model, device_ids=['cuda:0'])
state_dict = experiment.load_best_model(device='cuda:0')['model_state_dict']
model.load_state_dict(state_dict)
model.module.loss.debug_prints = True

batch_size = 5

# Eval for different noise levels
noise_summaries = {'noise_levels': noise_levels}
for noise in noise_levels:
    data_config.update({'point_noise_w': noise})

    if 'class' in data_config:
        data_class = getattr(data, data_config['class'])
        dataset = data_class(system_info['datasets_path'], data_config, gt_caching=True, feature_caching=True)
    else:
        dataset = data.GarmentStitchPairsDataset(
            system_info['datasets_path'], data_config, gt_caching=True, feature_caching=True)

    print(dataset.config)
    print('Batch: {}, Split: {}'.format(batch_size, split))

    datawrapper = data.DatasetWrapper(dataset, known_split=split, batch_size=batch_size)

    # ------- Evaluate --------

    loss = eval_metrics(model, datawrapper, 'test')
    print('Metrics on seen test set for noise {}: {}'.format(noise, loss))

    for key, value in loss.items():
        if key in noise_summaries:
            noise_summaries[key].append(value)
        else:
            noise_summaries[key] = [value]
    
print('Noise levels: ')
print(json.dumps(noise_summaries, sort_keys=False, indent=2))
with open(os.path.join(system_info['output'], 'seen_noise_levels_' + datetime.now().strftime('%y%m%d-%H-%M-%S') + '.json'), 'w') as f_json:
    json.dump(noise_summaries, f_json, sort_keys=False, indent=2)
