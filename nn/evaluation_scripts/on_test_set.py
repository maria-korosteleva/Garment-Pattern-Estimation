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

# --- Predict shape from shape experimnet ---
shape_datawrapper, shape_model, shape_experiment = load_experiment(
    'Filtered-att-data-condenced-classes', '390wuxbm', 
    in_batch_size=5, in_device='cuda:0')
prediction_path = shape_datawrapper.predict(
   shape_model, save_to=Path(system_info['output']), sections=['test'])   # 'validation', 

# prediction_path = Path('D:/GK-Pattern-Outputs/nn_pred_210825-04-14-21')

# --- Predict stitches for given prediction ---
# On test
data_folders = os.listdir(str(prediction_path / 'test'))
stitch_datawrapper, stitch_model, stitch_experiment = load_experiment(
    'Filtered-stitches-on-predss-split', 'a94r1949', in_device='cuda:0',
    in_data_folders=data_folders, in_datapath=prediction_path / 'test', 
    in_batch_size=1  # singletone batch to allow different number of edge pairs in different samples
)
# for all edge pairs
stitch_datawrapper.dataset.config.update(random_pairs_mode=False) 

# ------- Evaluate stitch prediction --------
loss = eval_metrics(stitch_model, stitch_datawrapper, 'full')
print('Sitch metrics: {}'.format(loss))
breakdown = eval_metrics(stitch_model, stitch_datawrapper, 'full_per_data_folder')
print('Sitch metrics per dataset: {}'.format(breakdown))

# only on examples with correctly predicted number of panels
corr_stitch_dataset = data.GarmentStitchPairsDataset(
    prediction_path / 'test', 
    stitch_datawrapper.dataset.config, 
    gt_caching=True, feature_caching=True, filter_correct_n_panels=True) 
corr_stitch_datawrapper = data.DatasetWrapper(corr_stitch_dataset, batch_size=1)

corr_metrics_values = eval_metrics(stitch_model, corr_stitch_datawrapper, 'full')
print('Sitch correct metrics: {}'.format(corr_metrics_values))
corr_breakdown = eval_metrics(stitch_model, corr_stitch_datawrapper, 'full_per_data_folder')
print('Stitch correct metrics per dataset: {}'.format(corr_breakdown))

# experiment.add_statistic('valid_on_best', valid_loss)
# experiment.add_statistic('valid', valid_breakdown)
stitch_experiment.add_statistic('test_preds_full', loss)
stitch_experiment.add_statistic('test_preds', breakdown)
stitch_experiment.add_statistic('test_corr_full', corr_metrics_values)
stitch_experiment.add_statistic('test_corr', corr_breakdown)
stitch_experiment.add_statistic('shape_model', 'Filtered-att-data-condenced-classes-390wuxbm')

# -------- Predict ---------
# save prediction for validation to file
prediction_path = stitch_datawrapper.predict(stitch_model, save_to=Path(system_info['output']), sections=['full'])
print('Saved to {}'.format(prediction_path))
# reflect predictions info in expetiment
stitch_experiment.add_statistic('test-pred_folder', prediction_path.name)

art_name = 'multi-data-test' if len(stitch_datawrapper.dataset.data_folders) > 1 else stitch_datawrapper.dataset.data_folders[0]  # + '-scan'
stitch_experiment.add_artifact(prediction_path, art_name, 'result')
