"""Evaluate a model on the data"""

from pathlib import Path
import torch
from torch import nn
import argparse

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys,inspect
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# My modules
import customconfig
import data
import metrics
from experiment import load_experiment

system_info = customconfig.Properties('./system.json')

dataset_list = [
    'jacket_hood_sleeveless_150',
    'skirt_waistband_150', 
    'tee_hood_150',
    'jacket_sleeveless_150',
    'dress_150',
    'jumpsuit_150',
    'wb_jumpsuit_sleeveless_150'
]

# ---Load data and models ---
shape_datawrapper, shape_model = load_experiment(
    'All-predefined-order-att-max', 's8fj6bqz', in_data_folders=dataset_list,
    in_batch_size=5, in_device='cuda:0')

# --- Predict shape from shape experimnet ---
prediction_path = shape_datawrapper.predict(
    shape_model, save_to=Path(system_info['output']), sections=['full'])


# --- Predict stitches for given prediction ---
data_folders = os.listdir(str(prediction_path / 'full'))
stitch_datawrapper, stitch_model = load_experiment(
    'All-stitches-800', '35515dwx', in_device='cuda:0',
    in_data_folders=data_folders, in_datapath=prediction_path / 'full', 
    in_batch_size=1  # singletone batch to allow different number of edge pairs in different samples
)
# for all edge pairs
stitch_datawrapper.dataset.config.update(random_pairs_mode=False)  

# ------- Evaluate stitch prediction --------
metrics = metrics.eval_metrics(stitch_model, stitch_datawrapper, 'full')
print('Sitch metrics: {}'.format(metrics))
breakdown = metrics.eval_metrics(stitch_model, stitch_datawrapper, 'full_per_data_folder')
print('Stitch metrics per dataset: {}'.format(breakdown))

# experiment.add_statistic('unseen_full', metrics)
# experiment.add_statistic('unseen', breakdown)
# experiment.add_statistic('unseen_folders', dataset_list)

# # -------- Predict ---------
# # save prediction for validation to file
# prediction_path = datawrapper.predict(model, save_to=Path(system_info['output']), sections=['validation', 'test'])
# print('Saved to {}'.format(prediction_path))
# # # reflect predictions info in expetiment
# experiment.add_statistic('pred_folder', prediction_path.name)

# art_name = 'multi-data' if len(datawrapper.dataset.data_folders) > 1 else datawrapper.dataset.data_folders[0]  # + '-scan'
# experiment.add_artifact(prediction_path, art_name, 'result')