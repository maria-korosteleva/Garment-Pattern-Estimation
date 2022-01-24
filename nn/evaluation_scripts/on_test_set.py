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
# shape_datawrapper, shape_model, shape_experiment = load_experiment('All-predefined-order-att-max', 's8fj6bqz', in_batch_size=5, in_device='cuda:0')
# prediction_path = shape_datawrapper.predict(
#    shape_model, save_to=Path(system_info['output']), sections=['validation', 'test'])

prediction_path = Path('D:/GK-Pattern-Outputs/nn_pred_210825-04-14-21')

# --- Predict stitches for given prediction ---
# On test
data_folders = os.listdir(str(prediction_path / 'test'))
stitch_datawrapper, stitch_model, stitch_experiment = load_experiment(
    'All-stitches-800', '35515dwx', in_device='cuda:0',
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

# experiment.add_statistic('valid_on_best', valid_loss)
# experiment.add_statistic('valid', valid_breakdown)
stitch_experiment.add_statistic('test_preds_full', loss)
stitch_experiment.add_statistic('test_preds', breakdown)
stitch_experiment.add_statistic('shape_model', 'All-predefined-order-att-max-s8fj6bqz')

# -------- Predict ---------
# save prediction for validation to file

# TODO is saves original svgs to the output folder, not the predicted patten!!!!
# prediction_path = datawrapper.predict(stitch_model, save_to=Path(system_info['output']), sections=['full'])
# print('Saved to {}'.format(prediction_path))
# reflect predictions info in expetiment
# experiment.add_statistic('pred_folder', prediction_path.name)

# art_name = 'multi-data' if len(datawrapper.dataset.data_folders) > 1 else datawrapper.dataset.data_folders[0]  # + '-scan'
# experiment.add_artifact(prediction_path, art_name, 'result')
