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

# TODO Move these routines to experiment class

def load_dataset(experiment, data_root, eval_config={}, batch_size=5):
    """Shortcut to load dataset
    
        NOTE: small default batch size for evaluation even on lightweights machines
    """

    # data_config also contains the names of datasets to use
    split, _, data_config = experiment.data_info()  # note that run is not initialized -- we use info from finished run

    # Extra evaluation configuration
    data_config.update(eval_config)

    # Dataset
    data_class = getattr(data, data_config['class'])
    dataset = data_class(data_root, data_config, gt_caching=True, feature_caching=True)

    
    datawrapper = data.DatasetWrapper(dataset, known_split=split, batch_size=batch_size)

    return dataset, datawrapper

def load_model(experiment, data_config):
    model_class = getattr(nets, experiment.NN_config()['model'])
    model = model_class(data_config, experiment.NN_config(), experiment.NN_config()['loss'])
    model = nn.DataParallel(model, device_ids=['cuda:0'])   # Assuming all models trained as DataParallel

    # Load model weights
    state_dict = experiment.load_best_model(device='cuda:0')['model_state_dict']
    model.load_state_dict(state_dict)
    model.module.loss.debug_prints = True

    return model

def prediction(save_to, experiment, model, datawrapper, nick='test', sections=['test'], art_name='multi-data'):
    # TODO better place for this routine? In experiment itself?
    prediction_path = datawrapper.predict(model, save_to=save_to, sections=sections)

    experiment.add_statistic(nick + '_folder', prediction_path.name, log='Prediction save path')

    art_name = art_name if len(datawrapper.dataset.data_folders) > 1 else datawrapper.dataset.data_folders[0]
    experiment.add_artifact(prediction_path, art_name, 'result')

    return prediction_path

if __name__ == "__main__":

    shape_config, stitch_config, args = get_values_from_args()
    system_info = customconfig.Properties('./system.json')

    # ----- Panel Shape predictions & evals --------
    if not args.pred_path:
        # from experiment
        shape_experiment = ExperimentWrappper(shape_config, system_info['wandb_username'])  # finished experiment
        if not shape_experiment.is_finished():
            print('Warning::Evaluating unfinished experiment')
        shape_dataset, shape_datawrapper = load_dataset(
            shape_experiment, system_info['datasets_path'],
            {'obj_filetag': 'sim', 'point_noise_w': 0})  # DEBUG -- example!
        shape_model = load_model(shape_experiment, shape_dataset.config)

        # Evaluate
        test_metrics = eval_metrics(shape_model, shape_datawrapper, 'test')
        test_breakdown = eval_metrics(shape_model, shape_datawrapper, 'test_per_data_folder')

        shape_experiment.add_statistic('test_on_best', test_metrics, log='Test metrics')
        shape_experiment.add_statistic('test', test_breakdown, log='Test metrics per dataset')

        if args.predict or stitch_config:  # requested or needed for stitch prediction
            # Predict 
            shape_prediction_path = prediction(
                Path(system_info['output']), shape_experiment, shape_model, shape_datawrapper, 
                nick='test_pred')

    # ----- With stitches -- predictions & evals --------
    if stitch_config:  # skip if stitch model config is not specified
        # Update data path
        in_datapath=shape_prediction_path / 'test'

        # Load
        stitch_experiment = ExperimentWrappper(stitch_config, system_info['wandb_username'])  # finished experiment
        if not stitch_experiment.is_finished():
            print('Warning::Evaluating unfinished experiment')
        stitch_dataset, stitch_datawrapper = load_dataset(
            stitch_experiment, shape_prediction_path / 'test', 
            batch_size=1)  # Num of edge pairs at test time is different for each sewing pattern 
        stitch_datawrapper.dataset.config.update(random_pairs_mode=False)  # use all edge pairs in evaluation
        stitch_model = load_model(stitch_experiment, stitch_dataset.config)

        # Evaluate stitch prediction
        loss = eval_metrics(stitch_model, stitch_datawrapper, 'full')
        breakdown = eval_metrics(stitch_model, stitch_datawrapper, 'full_per_data_folder')
        stitch_experiment.add_statistic('test_preds_full', loss, log='Sitch metrics')
        stitch_experiment.add_statistic('test_preds', breakdown, log='Sitch metrics per dataset')
        stitch_experiment.add_statistic('shape_model', shape_experiment.full_name())

        if args.predict:
            stitch_prediction_path = prediction(
                Path(system_info['output']), stitch_experiment, stitch_model, stitch_datawrapper, nick='test_pred')

        if args.correct_panels:
            # only on examples with correctly predicted number of panels
            corr_stitch_dataset = data.GarmentStitchPairsDataset(
                shape_prediction_path / 'test', stitch_datawrapper.dataset.config, 
                gt_caching=True, feature_caching=True, filter_correct_n_panels=True) 
            corr_stitch_datawrapper = data.DatasetWrapper(corr_stitch_dataset, batch_size=1)

            corr_metrics_values = eval_metrics(stitch_model, corr_stitch_datawrapper, 'full')
            corr_breakdown = eval_metrics(stitch_model, corr_stitch_datawrapper, 'full_per_data_folder')

            # Upload
            stitch_experiment.add_statistic('test_corr_full', corr_metrics_values, 'Metrics on correct patterns')
            stitch_experiment.add_statistic('test_corr', corr_breakdown, 'Metrics on correct patterns per dataset')