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
import nets
from trainer import Trainer
from experiment import WandbRunWrappper

def get_values_from_args():
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()
    
    # Default values from run 3cyu4gef, best accuracy\speed after sweep y1mmngej

    # basic
    parser.add_argument('--cluster_by', '-cb', help='', type=str, default='translation')
    parser.add_argument('--diff_cluster_threshold', '-d', help='', type=float, default=0.1)
    parser.add_argument('--version', '-v', help='Checkpoint version to request', type=int, default=-1)

    args = parser.parse_args()
    print(args)

    loss_config = {
        # 'epoch_with_order_matching': 0,
        # 'panel_origin_invariant_loss': False,
        # 'panel_order_inariant_loss': True,
        # 'order_by': 'shape_translation',   # placement, translation, stitches, shape_translation

        'cluster_by': args.cluster_by,  # 'panel_encodings', 'order_feature', 'translation'
        'epoch_with_cluster_checks': 100,
        'gap_cluster_threshold': 0.0,
        'diff_cluster_threshold': args.diff_cluster_threshold,  # testing New!!
        'cluster_gap_nrefs': 5,
        'cluster_with_singles': True,
        'cluster_memory_by_epoch': False,

        'loss_components': ['shape'],  # , 'loop', 'rotation', 'translation'],
        'quality_components': ['shape'],  #, 'discrete', 'rotation', 'translation'],
    }

    return loss_config, args.version


# --------------- from experimnet ---------
system_info = customconfig.Properties('./system.json')
experiment = WandbRunWrappper(
    system_info['wandb_username'],
    project_name='Garments-Reconstruction', 
    run_name='Tee-JS-stitches-all', 
    run_id='2hfx5dkv')  # finished experiment

if not experiment.is_finished():
    print('Warning::Evaluating unfinished experiment')

# -------- data -------
# data_config also contains the names of datasets to use
split, batch_size, data_config = experiment.data_info()  # note that run is not initialized -- we use info from finished run

data_config.update({'obj_filetag': 'sim'})  # scan imitation stats

if 'class' in data_config:
    data_class = getattr(data, data_config['class'])
    dataset = data_class(system_info['datasets_path'], data_config, gt_caching=True, feature_caching=True)
else:
    dataset = data.GarmentStitchPairsDataset(
        system_info['datasets_path'], data_config, gt_caching=True, feature_caching=True)

print(dataset.config)
print('Batch: {}, Split: {}'.format(batch_size, split))

# batch_size = 5

datawrapper = data.DatasetWrapper(dataset, known_split=split, batch_size=batch_size)

# From input
args_loss_config, checkpoint_version = get_values_from_args()

# DEBUG Loss config
loss_config = experiment.NN_config()['loss']
print(args_loss_config)
loss_config.update(args_loss_config)

# ----- Model architecture -----
model_class = getattr(nets, experiment.NN_config()['model'])
# model = nets.GarmentFullPattern3DDisentangle(dataset.config, experiment.NN_config(), experiment.NN_config()['loss'])
# model = nets.GarmentAttentivePattern3D(dataset.config, experiment.NN_config(), experiment.NN_config()['loss'])
model = model_class(dataset.config, experiment.NN_config(), loss_config)

if 'device_ids' in experiment.NN_config():  # model from multi-gpu training case
    model = nn.DataParallel(model, device_ids=['cuda:0'])

# state_dict = 
if checkpoint_version >= 0: 
    state_dict = experiment.load_checkpoint_file(version=checkpoint_version, device='cuda:0')['model_state_dict'] 
else:
    state_dict = experiment.load_best_model(device='cuda:0')['model_state_dict']

model.load_state_dict(state_dict)

model.module.loss.debug_prints = True

# ------- Evaluate --------
valid_loss = metrics.eval_metrics(model, datawrapper, 'validation')
print('Validation metrics: {}'.format(valid_loss))
# valid_breakdown = metrics.eval_metrics(model, datawrapper, 'valid_per_data_folder')
# print('Validation metrics per dataset: {}'.format(valid_breakdown))

# test_metrics = metrics.eval_metrics(model, datawrapper, 'test')
# print('Test metrics: {}'.format(test_metrics))
# test_breakdown = metrics.eval_metrics(model, datawrapper, 'test_per_data_folder')
# print('Test metrics per dataset: {}'.format(test_breakdown))

# experiment.add_statistic('valid_on_best', valid_loss)
# experiment.add_statistic('valid', valid_breakdown)
# experiment.add_statistic('test_on_best', test_metrics)
# experiment.add_statistic('test', test_breakdown)

# # -------- Predict ---------
# # save prediction for validation to file
# prediction_path = datawrapper.predict(model, save_to=Path(system_info['output']), sections=['validation', 'test'])
# print('Saved to {}'.format(prediction_path))
# # # reflect predictions info in expetiment
# experiment.add_statistic('pred_folder', prediction_path.name)

# art_name = 'multi-data' if len(datawrapper.dataset.data_folders) > 1 else datawrapper.dataset.data_folders[0]  # + '-scan'
# experiment.add_artifact(prediction_path, art_name, 'result')
