"""Evaluate a model on the data"""

from pathlib import Path
import torch
import torch.nn as nn
import json
from datetime import datetime

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
noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]  # []   # 

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


# ----- Model architecture -- load once -----
model_class = getattr(nets, experiment.NN_config()['model'])
model = model_class(data_config, experiment.NN_config(), experiment.NN_config()['loss'])

if 'device_ids' in experiment.NN_config():  # model from multi-gpu training case
    model = nn.DataParallel(model, device_ids=['cuda:0'])
model.load_state_dict(experiment.load_best_model(device='cuda:0')['model_state_dict'])

print(data_config['class'])

noise_summaries = {'noise_levels': noise_levels}

for noise in noise_levels:
    # Force noise
    data_config.update({'point_noise_w': noise})

    data_class = getattr(data, data_config['class'])
    dataset = data_class(system_info['datasets_path'] + '/test', data_config, gt_caching=True, feature_caching=True)

    datawrapper = data.DatasetWrapper(dataset, batch_size=batch_size)  # NOTE no split given -- evaluating on the full loaded dataset!!


    # ------- Evaluate --------
    loss = eval_metrics(model, datawrapper, 'full')
    print('Metrics on unseen set for noise {}: {}'.format(noise, loss))

    for key, value in loss.items():
        if key in noise_summaries:
            noise_summaries[key].append(value)
        else:
            noise_summaries[key] = [value]
    
print('Noise levels: ')
print(json.dumps(noise_summaries, sort_keys=False, indent=2))
with open(os.path.join(system_info['output'], 'unseen_noise_levels_' + datetime.now().strftime('%y%m%d-%H-%M-%S') + '.json'), 'w') as f_json:
    json.dump(noise_summaries, f_json, sort_keys=False, indent=2)

    