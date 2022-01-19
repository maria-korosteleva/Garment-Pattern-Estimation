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
from metrics.eval_utils import eval_metrics
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

# --- Predict shape from shape experimnet ---
shape_datawrapper, shape_model, _ = load_experiment(
    'All-predefined-order-att-max', 's8fj6bqz', 
    in_data_folders=dataset_list, in_datapath=Path(system_info['datasets_path']) / 'test',
    in_batch_size=5, in_device='cuda:0')
prediction_path = shape_datawrapper.predict(
    shape_model, save_to=Path(system_info['output']), sections=['full'])

# prediction_path = Path('D:/GK-Pattern-Outputs/nn_pred_210824-03-28-38')

# --- Predict stitches for given prediction ---
data_folders = os.listdir(str(prediction_path / 'full'))
stitch_datawrapper, stitch_model, stitch_experiment = load_experiment(
    'All-stitches-800', '35515dwx', in_device='cuda:0',
    in_data_folders=data_folders, in_datapath=prediction_path / 'full', 
    in_batch_size=1  # singletone batch to allow different number of edge pairs in different samples
)
# for all edge pairs
stitch_datawrapper.dataset.config.update(random_pairs_mode=False)  

# ------- Evaluate stitch prediction --------
metrics_values = eval_metrics(stitch_model, stitch_datawrapper, 'full')
print('Sitch metrics: {}'.format(metrics_values))
breakdown = eval_metrics(stitch_model, stitch_datawrapper, 'full_per_data_folder')
print('Stitch metrics per dataset: {}'.format(breakdown))

# only on examples with correctly predicted number of panels
corr_stitch_dataset = data.GarmentStitchPairsDataset(
    prediction_path / 'full', 
    stitch_datawrapper.dataset.config, 
    gt_caching=True, feature_caching=True, filter_correct_n_panels=True)
split, batch_size, _ = stitch_experiment.data_info()  
corr_stitch_datawrapper = data.DatasetWrapper(corr_stitch_dataset, known_split=split, batch_size=batch_size)

corr_metrics_values = eval_metrics(stitch_model, corr_stitch_datawrapper, 'full')
print('Sitch correct metrics: {}'.format(metrics_values))
corr_breakdown = eval_metrics(stitch_model, corr_stitch_datawrapper, 'full_per_data_folder')
print('Stitch correct metrics per dataset: {}'.format(breakdown))

stitch_experiment.add_statistic('unseen_preds_full', metrics_values)
stitch_experiment.add_statistic('unseen_preds', breakdown)
stitch_experiment.add_statistic('unseen_corr_preds_full', corr_metrics_values)
stitch_experiment.add_statistic('unseen_corr_preds', corr_breakdown)
stitch_experiment.add_statistic('unseen_shape_model', 'All-predefined-order-att-max-s8fj6bqz')

# # -------- Predict ---------
# # save prediction for validation to file
prediction_path = stitch_datawrapper.predict(stitch_model, save_to=Path(system_info['output']), sections=['full'])
print('Saved to {}'.format(prediction_path))
# # # reflect predictions info in expetiment
# experiment.add_statistic('pred_folder', prediction_path.name)

# art_name = 'multi-data' if len(datawrapper.dataset.data_folders) > 1 else datawrapper.dataset.data_folders[0]  # + '-scan'
# experiment.add_artifact(prediction_path, art_name, 'result')