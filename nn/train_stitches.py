from pathlib import Path
import argparse
import numpy as np
import torch.nn as nn
import os
from distutils import dir_util

# My modules
import customconfig
import data
import nets
from metrics.eval_utils import eval_metrics
from trainer import Trainer
from experiment import WandbRunWrappper, load_experiment
import nn.evaluation_scripts.latent_space_vis as tsne_plot

import warnings
warnings.filterwarnings('ignore')  # , category='UserWarning'


def get_default_values():
    """Controlling the setup"""

    data_config = {
        'stitched_edge_pairs_num': 200,
        'non_stitched_edge_pairs_num': 200,
        'shuffle_pairs': True, 
        'shuffle_pairs_order': True
    }

    nn_config = {
        'stitch_hidden_size': 200, 
        'stitch_mlp_n_layers': 3
    }

    loss_config = {}

    return data_config, nn_config, loss_config, args.net_seed


def get_data_config(in_config, old_stats=False):
    """Shortcut to control data configuration
        Note that the old experiment is HARDCODED!!!!!"""
    if old_stats:
        # get data stats from older runs to save runtime
        old_experiment = WandbRunWrappper(
            system_info['wandb_username'],
            project_name='Garments-Reconstruction', 
            run_name='All-predefined-order-att-max', run_id='s8fj6bqz'  # all data 800
        )
        # NOTE data stats are ONLY correct for a specific data split, so these two need to go together
        split, _, data_config = old_experiment.data_info()

    else:  # default split for reproducibility
        # NOTE addining 'filename' property to the split will force the data to be loaded from that list, instead of being randomly generated
        split = {'valid_per_type': 150, 'test_per_type': 150, 'random_seed': 10, 'type': 'count', 'filename': './wandb/data_split.json'} 
        data_config = {}  

    # update with freshly configured values
    data_config.update(in_config)

    return split, data_config


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

    system_info = customconfig.Properties('./system.json')

    # Get training data from the shape experiment!
    shape_datawrapper, shape_model, shape_experiment = load_experiment(
        'Filtered-att-data-condenced-classes', '390wuxbm', in_batch_size=60, in_device='cuda:0')
    prediction_path = shape_datawrapper.predict(
        shape_model, save_to=Path(system_info['output']), sections=['train', 'validation', 'test'])
    # merge into one repo
    data_path = merge_repos(prediction_path, ['train', 'validation', 'test'])
    dataset_list = os.listdir(str(data_path))

    # -- Setup stitch experiment --
    experiment = WandbRunWrappper(
        system_info['wandb_username'], 
        project_name='Garments-Reconstruction', 
        run_name='Filtered-stitches-on-predss-split', 
        run_id=None, no_sync=False)   # set run id to resume unfinished run!

    # NOTE this dataset involves point sampling SO data stats from previous runs might not be correct, especially if we change the number of samples
    in_data_config, in_nn_config, in_loss_config, net_seed = get_default_values()
    _, data_config = get_data_config(in_data_config, old_stats=False)  # DEBUG
    split, _, _ = shape_experiment.data_info()

    data_config.update(data_folders=dataset_list)
    dataset = data.GarmentStitchPairsDataset(data_path, data_config, gt_caching=True, feature_caching=True)

    # -- Training --
    # No split provided -- use the whole data!!
    # TRY with split
    trainer = Trainer(experiment, dataset, data_split=split, with_norm=True, with_visualization=False)
    trainer.init_randomizer(net_seed)
    model = nets.StitchOnEdge3DPairs(dataset.config, in_nn_config, in_loss_config)

    # Multi-GPU!!!
    model = nn.DataParallel(model, device_ids=['cuda:0'])
    model.module.config['device_ids'] = model.device_ids

    model.module.loss.with_quality_eval = True  # False to save compute time
    model.module.loss.debug_prints = True  # False to avoid extra prints
    if hasattr(model.module, 'config'):
        trainer.update_config(NN=model.module.config)  # save NN configuration

    trainer.fit(model)  # Magic happens here

    # --------------- Final evaluation -- same as in test.py --------------
    # On the best-performing model
    try:
        model.load_state_dict(experiment.load_best_model()['model_state_dict'])
    except BaseException as e:  # not the best to catch all the exceptions here, but should work for most of cases foe now
        print(e)
        print('Train::Warning::Proceeding to evaluation with the current (final) model state')

    datawrapper = trainer.datawraper

    final_metrics = eval_metrics(model, datawrapper, 'validation')
    print('Validation metrics: {}'.format(final_metrics))
    experiment.add_statistic('valid_on_best', final_metrics)

    final_metrics = eval_metrics(model, datawrapper, 'valid_per_data_folder')
    print('Validation metrics breakdown: {}'.format(final_metrics))
    experiment.add_statistic('valid', final_metrics)

    final_metrics = eval_metrics(model, datawrapper, 'test')
    print('Test metrics: {}'.format(final_metrics))
    experiment.add_statistic('test_on_best', final_metrics)

    final_metrics = eval_metrics(model, datawrapper, 'test_per_data_folder')
    print('Test metrics breakdown: {}'.format(final_metrics))
    experiment.add_statistic('test', final_metrics)
