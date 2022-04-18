"""Evaluate a model on the data"""
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
    parser.add_argument('--config', '-c', help='YAML configuration file', type=str, default='./models/att/att.yaml')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_dataset(experiment, eval_config):
    # data_config also contains the names of datasets to use
    split, _, data_config = experiment.data_info()  # note that run is not initialized -- we use info from finished run

    # Extra evaluation configuration
    data_config.update(eval_config)

    # Dataset
    data_class = getattr(data, data_config['class'])
    dataset = data_class(system_info['datasets_path'], data_config, gt_caching=True, feature_caching=True)

    # small batch size for evaluation even on lightweights machines
    datawrapper = data.DatasetWrapper(dataset, known_split=split, batch_size=5)

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


if __name__ == "__main__":

    config = get_values_from_args()
    system_info = customconfig.Properties('./system.json')

    # from experiment
    experiment = ExperimentWrappper(config, system_info['wandb_username'])  # finished experiment
    if not experiment.is_finished():
        print('Warning::Evaluating unfinished experiment')

    # -------- data -------
    dataset, datawrapper = load_dataset(
        experiment, 
        {'obj_filetag': 'sim', 'point_noise_w': 0})  # DEBUG -- example!

    # ----- Model architecture -----
    model = load_model(experiment, dataset.config)

    # ------- Evaluate --------
    test_metrics = eval_metrics(model, datawrapper, 'test')
    print('Test metrics: {}'.format(json.dumps(test_metrics, sort_keys=True, indent=2)))
    test_breakdown = eval_metrics(model, datawrapper, 'test_per_data_folder')
    print('Test metrics per dataset: {}'.format(json.dumps(test_breakdown, sort_keys=True, indent=2)))

    experiment.add_statistic('test_on_best', test_metrics)
    experiment.add_statistic('test', test_breakdown)

    # -------- Predict ---------
    # save prediction for validation to file
    # prediction_path = datawrapper.predict(model, save_to=Path(system_info['output']), sections=['validation', 'test'])
    # print('Saved to {}'.format(prediction_path))
    # # # reflect predictions info in expetiment
    # experiment.add_statistic('test_pred_folder', prediction_path.name)

    # art_name = 'multi-data' if len(datawrapper.dataset.data_folders) > 1 else datawrapper.dataset.data_folders[0]  # + '-scan'
    # experiment.add_artifact(prediction_path, art_name, 'result')
