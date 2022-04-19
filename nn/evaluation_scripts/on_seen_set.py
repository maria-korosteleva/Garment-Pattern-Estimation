"""Evaluate Panel Shape prediction model or full NeuralTailor framework on the seen garment types

    Outputs the resulting values to stdout and writes them to the wandb run (if available)
"""

from pathlib import Path
from torch import nn
import argparse
import yaml
import json

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# My modules
import customconfig
import data
from metrics.eval_utils import eval_metrics
import nets
from experiment import ExperimentWrappper


def get_values_from_args():
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()
    
    # basic
    parser.add_argument('-sh', '--shape_config', help='YAML configuration file', type=str, default='./models/att/att.yaml')
    parser.add_argument('-st', '--stitch_config', help='YAML configuration file', type=str, default='')  # evaluates only the shape model by default
    parser.add_argument(
        '--pred_path', type=str, default='',
        help='Path to the sewing pattern prediction that can be used for stitch evaluations. Set to skip loading, eval & prediction on Pattern Shape module')

    parser.add_argument('-p', '--predict', help='if set, saves sewing pattern predictions to output folder and uploads them to the run', action='store_true')
    parser.add_argument('-corr', '--correct_panels', help='if set, additionally evaluate stitch information only on the patterns with correctly predicted # of panels', action='store_true')

    args = parser.parse_args()
    with open(args.shape_config, 'r') as f:
        shape_config = yaml.safe_load(f)
    
    if args.stitch_config:
        with open(args.stitch_config, 'r') as f:
            stitch_config = yaml.safe_load(f)
    else:
        stitch_config = None

    # DEBUG
    print(f'Command line args: {args}')

    return shape_config, stitch_config, args


if __name__ == "__main__":

    shape_config, stitch_config, args = get_values_from_args()
    system_info = customconfig.Properties('./system.json')

    # ----- Panel Shape predictions & evals --------
    if not args.pred_path:
        # from experiment
        shape_experiment = ExperimentWrappper(shape_config, system_info['wandb_username'])  # finished experiment
        if not shape_experiment.is_finished():
            print('Warning::Evaluating unfinished experiment')
        shape_dataset, shape_datawrapper = shape_experiment.load_dataset(
            system_info['datasets_path'],
            {'obj_filetag': 'sim', 'point_noise_w': 0})  # DEBUG -- example!
        shape_model = shape_experiment.load_model(shape_dataset.config)

        # Evaluate
        test_metrics = eval_metrics(shape_model, shape_datawrapper, 'test')
        test_breakdown = eval_metrics(shape_model, shape_datawrapper, 'test_per_data_folder')

        shape_experiment.add_statistic('test_on_best', test_metrics, log='Test metrics')
        shape_experiment.add_statistic('test', test_breakdown, log='Test metrics per dataset')

        if args.predict or stitch_config:  # requested or needed for stitch prediction
            # Predict 
            shape_prediction_path = shape_experiment.prediction(
                Path(system_info['output']), shape_model, shape_datawrapper, 
                nick='test_pred')

    # ----- With stitches -- predictions & evals --------
    if stitch_config:  # skip if stitch model config is not specified
        # Update data path
        in_datapath = Path(args.pred_path) / 'test' if args.pred_path else shape_prediction_path / 'test'

        # Load
        stitch_experiment = ExperimentWrappper(stitch_config, system_info['wandb_username'])  # finished experiment
        if not stitch_experiment.is_finished():
            print('Warning::Evaluating unfinished experiment')
        stitch_dataset, stitch_datawrapper = stitch_experiment.load_dataset(in_datapath, batch_size=1, load_all=True)  # Num of edge pairs at test time is different for each sewing pattern 
        stitch_datawrapper.dataset.config.update(random_pairs_mode=False)  # use all edge pairs in evaluation
        stitch_model =stitch_experiment.load_model(stitch_dataset.config)

        # Evaluate stitch prediction
        loss = eval_metrics(stitch_model, stitch_datawrapper, 'full')
        breakdown = eval_metrics(stitch_model, stitch_datawrapper, 'full_per_data_folder')
        stitch_experiment.add_statistic('test_preds_full', loss, log='Sitch metrics')
        stitch_experiment.add_statistic('test_preds', breakdown, log='Sitch metrics per dataset')
        try:
            stitch_experiment.add_statistic('shape_model', shape_experiment.full_name())
        except NameError as e:
            pass

        if args.predict:
            stitch_prediction_path = stitch_experiment.prediction(
                Path(system_info['output']), stitch_model, stitch_datawrapper, nick='test_pred')

        if args.correct_panels:
            # only on examples with correctly predicted number of panels
            corr_stitch_dataset = data.GarmentStitchPairsDataset(
                in_datapath, stitch_datawrapper.dataset.config, 
                gt_caching=True, feature_caching=True, filter_correct_n_panels=True) 
            corr_stitch_datawrapper = data.DatasetWrapper(corr_stitch_dataset, batch_size=1)

            corr_metrics_values = eval_metrics(stitch_model, corr_stitch_datawrapper, 'full')
            corr_breakdown = eval_metrics(stitch_model, corr_stitch_datawrapper, 'full_per_data_folder')

            # Upload
            stitch_experiment.add_statistic('test_corr_full', corr_metrics_values, 'Metrics on correct patterns')
            stitch_experiment.add_statistic('test_corr', corr_breakdown, 'Metrics on correct patterns per dataset')