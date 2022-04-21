"""Evaluate a model on the data"""

from pathlib import Path
import argparse
import json
from datetime import datetime
import yaml

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# My modules
import customconfig
from metrics.eval_utils import eval_metrics
from experiment import ExperimentWrappper

noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]  

def get_values_from_args():
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-sh', '--shape_config', help='YAML configuration file', type=str, default='./models/att/att.yaml')
    # Test set subset
    parser.add_argument('-u', '--unseen', help='if set, evaluates performance on unseen garment types, otherwise uses seen subset', action='store_true')

    args = parser.parse_args()
    with open(args.shape_config, 'r') as f:
        shape_config = yaml.safe_load(f)

    print(f'Command line args: {args}')

    return shape_config, args

if __name__ == "__main__":

    shape_config, args = get_values_from_args()
    system_info = customconfig.Properties('./system.json')
    tag = 'seen' if not args.unseen else 'unseen'

    shape_experiment = ExperimentWrappper(shape_config, system_info['wandb_username'])  # finished experiment
    shape_model = shape_experiment.load_model()

    # Eval for different noise levels
    noise_summaries = {'noise_levels': noise_levels}
    for noise in noise_levels:
        shape_dataset, shape_datawrapper = shape_experiment.load_dataset(
            Path(system_info['datasets_path']) / 'test' if args.unseen else system_info['datasets_path'],   # assuming dataset root structure
            {'point_noise_w': noise}, unseen=args.unseen)  

        # ------- Evaluate --------
        loss = eval_metrics(shape_model, shape_datawrapper, 'full' if args.unseen else 'test')
        print(f'Metrics on {tag} test set for noise {noise}: {loss}')
        for key, value in loss.items():
            if key in noise_summaries:
                noise_summaries[key].append(value)
            else:
                noise_summaries[key] = [value]
        
    print('Noise levels: ')
    print(json.dumps(noise_summaries, sort_keys=False, indent=2))
    with open(os.path.join(system_info['output'], f'{tag}_noise_levels_{datetime.now().strftime("%y%m%d-%H-%M-%S")}.json'), 'w') as f_json:
        json.dump(noise_summaries, f_json, sort_keys=False, indent=2)
