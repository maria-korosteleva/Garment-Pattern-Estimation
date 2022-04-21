"""Predicting a 2D pattern for the given 3D point clouds of garments -- 
    not necessarily from the garment dataset of this project

    NOTE: the point cloud files are expected to be just the .txt files 
          with the first three numbers in every line containing three world coordinates for a point.
"""

import argparse
from datetime import datetime
import numpy as np
from pathlib import Path
import torch
import traceback
import yaml

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# My modules
import customconfig
import data
from experiment import ExperimentWrappper
from pattern.wrappers import VisPattern


def get_values_from_args():
    """command line arguments to get a path to geometry file with a garment or a folder with OBJ files"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    system_info = customconfig.Properties('./system.json')
    parser = argparse.ArgumentParser()

    parser.add_argument('-sh', '--shape_config', help='YAML configuration file', type=str, default='./models/att/att.yaml')
    parser.add_argument('-st', '--stitch_config', help='YAML configuration file', type=str, default='./models/att/stitch_model.yaml') 

    parser.add_argument(
        '--file', '-f', help='Path to a garment point cloud file (.txt)', type=str, 
        default=None) 
    parser.add_argument(
        '--directory', '-dir', help='Path to a directory with point cloud files (.txt) to evaluate on', type=str, 
        default=None)
    parser.add_argument(
        '--save_tag', '-s', help='Tag the output directory name with this str', type=str, 
        default='per_sample')

    args = parser.parse_args()
    print(args)

    # load expriment configs
    args = parser.parse_args()
    with open(args.shape_config, 'r') as f:
        shape_config = yaml.safe_load(f)
    
    if args.stitch_config:
        with open(args.stitch_config, 'r') as f:
            stitch_config = yaml.safe_load(f)
    else:
        stitch_config = None

    # turn arguments into the list of obj files
    paths_list = []
    if args.file is None and args.directory is None: 
        # default value if no arguments provided
        raise ValueError('No inputs point cloud samples are provided')
    else:
        if args.file is not None:
            paths_list.append(Path(args.file))
        if args.directory is not None:
            directory = Path(args.directory)
            for elem in directory.glob('*'):
                if elem.is_file() and '.txt' in str(elem):
                    paths_list.append(elem)

    saving_path = Path(system_info['output']) / (args.save_tag + '_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
    saving_path.mkdir(parents=True)

    return shape_config, stitch_config, paths_list, saving_path




if __name__ == "__main__":
    
    system_info = customconfig.Properties('./system.json')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    shape_config, stitch_config, sample_paths, save_to = get_values_from_args()

    # --------------- Experiment to evaluate on ---------
    shape_experiment = ExperimentWrappper(shape_config, system_info['wandb_username'])
    stitch_experiment = ExperimentWrappper(stitch_config, system_info['wandb_username'])
    if not shape_experiment.is_finished():
        print('Warning::Evaluating unfinished experiment')

    # data stats from training 
    _, _, data_config = shape_experiment.data_info()  # need to get data stats

    # ------ prepare input data & construct batch -------
    points_list = []
    for filename in sample_paths:
        with open(filename, 'r') as pc_file: 
            points = []
            for line in pc_file:
                coords = [float(x) for x in line.split()]
                coords = coords[:3]
                points.append(coords)
        points = np.array(points)

        if abs(points.shape[0] - data_config['mesh_samples']) > 10:  # some tolerance to error
            selection = np.random.permutation(points.shape[0])[:data_config['mesh_samples']]
            print('Warning::Input point cloud has {} points while {} are expected. Needed #points was sampled'.format(
                points.shape[0], data_config['mesh_samples']))
            points = points[selection]
           
        if 'standardize' in data_config:
            points = (points - data_config['standardize']['f_shift']) / data_config['standardize']['f_scale']
        points_list.append(torch.tensor(points).float())

    # ----- Model (Pattern Shape) architecture -----
    shape_model = shape_experiment.load_model()
    shape_model.eval()

    # -------- Predict Shape ---------
    with torch.no_grad():
        points_batch = torch.stack(points_list).to(device)
        predictions = shape_model(points_batch)

    # ---- save shapes ----
    saving_path = save_to / 'shape'
    saving_path.mkdir(parents=True, exist_ok=True)
    names = [VisPattern.name_from_path(elem) for elem in sample_paths]
    data.save_garments_prediction(predictions, saving_path, data_config, names)


    # ========== Stitch prediction =========

    # ----- Model (Stitch Prediction) ------
    _, _, stitch_data_config = stitch_experiment.data_info()  # need to get data stats
    stitch_model = stitch_experiment.load_model()
    stitch_model.eval()

    # ----- predict & save stitches ------
    saving_path = save_to / 'stitched'
    saving_path.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(names):
        # "unbatch" dictionary
        prediction = {}
        for key in predictions:
            prediction[key] = predictions[key][idx]

        if data_config is not None and 'standardize' in data_config:
            # undo standardization  (outside of generinc conversion function due to custom std structure)
            gt_shifts = data_config['standardize']['gt_shift']
            gt_scales = data_config['standardize']['gt_scale']
            for key in gt_shifts:
                if key == 'stitch_tags' and not data_config['explicit_stitch_tags']:  
                    # ignore stitch tags update if explicit tags were not used
                    continue
                prediction[key] = prediction[key].cpu().numpy() * gt_scales[key] + gt_shifts[key]

        pattern = data.NNSewingPattern(view_ids=False)
        pattern.name = name
        try:
            pattern.pattern_from_tensors(
                prediction['outlines'], prediction['rotations'], prediction['translations'], 
                padded=True)   
            pattern.stitches_from_pair_classifier(stitch_model, stitch_data_config['standardize'])
            pattern.serialize(save_to / 'stitched', to_subfolder=True)

        except (RuntimeError, data.InvalidPatternDefError, TypeError) as e:
            print(traceback.format_exc())
            print(e)
            print('Saving predictions::Skipping pattern {}'.format(name))
            pass
