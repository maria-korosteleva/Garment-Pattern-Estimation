from distutils import dir_util
from pathlib import Path
import argparse
import numpy as np
import torch.nn as nn
import yaml

# My modules
import customconfig
import data
import nets
from metrics.eval_utils import eval_metrics
from trainer import Trainer
from experiment import ExperimentWrappper, load_experiment
import nn.evaluation_scripts.latent_space_vis as tsne_plot

import warnings
warnings.filterwarnings('ignore')  # , category='UserWarning'


def get_values_from_args():
    """command line arguments to control the run for running wandb Sweeps!"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', help='YAML configuration file', type=str, default='./models/att/att.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config 


def get_old_data_config(in_config):
    """Shortcut to control data configuration
        Note that the old experiment is HARDCODED!!!!!"""
    # get data stats from older runs to save runtime
    old_experiment = ExperimentWrappper(
        system_info['wandb_username'],
        project_name=in_config['old_experiment']['project_name'],
        run_name=in_config['old_experiment']['run_name'],
        run_id=in_config['old_experiment']['run_id']
    )
    # NOTE data stats are ONLY correct for a specific data split, so these two need to go together
    split, _, data_config = old_experiment.data_info()
    data_config = {
        'standardize': data_config['standardize'],
        'max_pattern_len': data_config['max_pattern_len'],
        'max_panel_len': data_config['max_panel_len'],
        'max_num_stitches': data_config['max_num_stitches'],  # the rest of the info is not needed here
        'max_datapoints_per_type': data_config['max_datapoints_per_type'] if 'max_datapoints_per_type' in data_config else None,  # keep the numbers too
        'panel_classification': data_config['panel_classification'],
        'filter_by_params': data_config['filter_by_params'],
        'mesh_samples': data_config['mesh_samples'],
        'obj_filetag': data_config['obj_filetag'],
        'point_noise_w': data_config['point_noise_w'] if 'point_noise_w' in data_config else 0
    }
    # update with freshly configured values
    in_config.update(data_config)
    
    print(split)

    return split, in_config


def merge_repos(root, repos):
    """ Create repository that merges the top ones"""

    root = Path(root)
    merge_target = root / 'merged'
    merge_target.mkdir(exist_ok=True)

    for repo in repos:
        dir_util.copy_tree(str(root / repo), str(merge_target))
    
    return merge_target


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)  # for readability

    config = get_values_from_args()

    system_info = customconfig.Properties('./system.json')
    
    experiment = ExperimentWrappper(
        config, # set run id in cofig to resume unfinished run!
        system_info['wandb_username'],
        no_sync=False)   

    # --- Data ---
    if 'old_experiment' in config['dataset'] and config['dataset']['old_experiment']['predictions']:
        # Use predictions of model from specified experiment as a dataset for training
        info = config['dataset']['old_experiment']
        shape_datawrapper, shape_model, shape_experiment = load_experiment(
            info['run_name'], info['run_id'], project=info['project_name'], 
            in_batch_size=config['trainer']['batch_size'], in_device=config['trainer']['devices'][0])
        prediction_path = shape_datawrapper.predict(
            shape_model, save_to=Path(system_info['output']), sections=['train', 'validation', 'test'], orig_folder_names=True)
        system_info['datasets_path'] = merge_repos(prediction_path, ['train', 'validation', 'test'])

    if 'old_experiment' in config['dataset'] and config['dataset']['old_experiment']['stats']:
        config['data_split'], config['dataset'] = get_old_data_config(config['dataset'])

    # Dataset Class
    data_class = getattr(data, config['dataset']['class'])
    dataset = data_class(Path(system_info['datasets_path']), config['dataset'], gt_caching=True, feature_caching=True)

    # --- Trainer --- 
    trainer = Trainer(
        config['trainer'], experiment, dataset, config['data_split'], 
        with_norm=True, with_visualization=config['trainer']['with_visualization'])  # only turn on visuals on custom garment data

    # --- Model ---
    trainer.init_randomizer()
    model_class = getattr(nets, config['NN']['model'])
    model = model_class(dataset.config, config['NN'], config['NN']['loss'])

    # Multi-GPU!!!
    model = nn.DataParallel(model, device_ids=config['trainer']['devices'])
    model.module.config['device_ids'] = model.device_ids

    print(f'Using devices: {model.device_ids}')

    model.module.loss.with_quality_eval = True  # False to save compute time
    model.module.loss.debug_prints = True  # False to avoid extra prints

    # --- TRAIN --- 
    trainer.fit(model)  # Magic happens here

    # --- Final evaluation ----
    # On the best-performing model
    try:
        model.load_state_dict(experiment.get_best_model()['model_state_dict'])
    except BaseException as e:  # not the best to catch all the exceptions here, but should work for most of cases foe now
        print(e)
        print('Train::Warning::Proceeding to evaluation with the current (final) model state')

    datawrapper = trainer.datawraper

    final_metrics = eval_metrics(model, datawrapper, 'validation')
    experiment.add_statistic('valid_on_best', final_metrics, log='Validation metrics')
    final_metrics = eval_metrics(model, datawrapper, 'valid_per_data_folder')
    experiment.add_statistic('valid', final_metrics, log='Validation metrics breakdown')
    final_metrics = eval_metrics(model, datawrapper, 'test')
    experiment.add_statistic('test_on_best', final_metrics, log='Test metrics')
    final_metrics = eval_metrics(model, datawrapper, 'test_per_data_folder')
    experiment.add_statistic('test', final_metrics, 'Test metrics breakdown')
