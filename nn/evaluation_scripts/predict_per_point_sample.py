"""Predicting a 2D pattern for the given 3D models of garments -- not necessarily from the garment datasets of this project"""

import argparse
from datetime import datetime
import igl
import numpy as np
from pathlib import Path
import shutil
import torch
import torch.nn as nn
import traceback

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys,inspect

from nn.pattern_converter import NNSewingPattern
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# My modules
import customconfig, nets, data
from experiment import ExperimentWrappper
from pattern.wrappers import VisPattern
from pattern_converter import NNSewingPattern, InvalidPatternDefError


def get_meshes_from_args():
    """command line arguments to get a path to geometry file with a garment or a folder with OBJ files"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    system_info = customconfig.Properties('./system.json')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file', '-f', help='Path to a garment geometry file', type=str, 
        default=None) 
    parser.add_argument(
        '--directory', '-dir', help='Path to a directory with geometry files to evaluate on', type=str, 
        default=None)
    parser.add_argument(
        '--save_tag', '-s', help='Tag the output directory name with this str', type=str, 
        default='per_sample')

    args = parser.parse_args()
    print(args)

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

    return paths_list, saving_path




if __name__ == "__main__":
    
    system_info = customconfig.Properties('./system.json')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sample_paths, save_to = get_meshes_from_args()

    # --------------- Experiment to evaluate on ---------
    shape_experiment = ExperimentWrappper(
        system_info['wandb_username'],
        project_name='Garments-Reconstruction', 
        run_name='RNN-no-stitch-5000-filt-cond', 
        run_id='3857jk4g')  # finished experiment
    stitch_experiment = ExperimentWrappper(
        system_info['wandb_username'],
        project_name='Garments-Reconstruction', 
        run_name='Filtered-stitches-on-RNN', 
        run_id='3ncgkjnh')  # finished experiment
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
            # raise ValueError('Input point cloud has {} points while {} are expected'.format(points.shape[0], data_config['mesh_samples']))
            print(points.shape)
           

        if 'standardize' in data_config:
            points = (points - data_config['standardize']['f_shift']) / data_config['standardize']['f_scale']
        points_list.append(torch.tensor(points).float())

    # ----- Model (Pattern Shape) architecture -----
    model_class = getattr(nets, shape_experiment.NN_config()['model'])
    model = model_class(data_config, shape_experiment.NN_config(), shape_experiment.NN_config()['loss'])
    if 'device_ids' in shape_experiment.NN_config():  # model from multi-gpu training case
        model = nn.DataParallel(model, device_ids=['cuda:0'])
    model.load_state_dict(shape_experiment.load_best_model()['model_state_dict'])
    model = model.to(device=device)
    model.eval()

    # -------- Predict Shape ---------
    with torch.no_grad():
        points_batch = torch.stack(points_list).to(device)
        predictions = model(points_batch)

    # ---- save shapes ----
    saving_path = save_to / 'shape'
    saving_path.mkdir(parents=True, exist_ok=True)

    names = [VisPattern.name_from_path(elem) for elem in sample_paths]
    data.save_garments_prediction(predictions, saving_path, data_config, names)


    # ========== Stitch prediction =========

    # ----- Model (Stitch Prediction) ------
    _, _, stitch_data_config = stitch_experiment.data_info()  # need to get data stats
    model_class = getattr(nets, stitch_experiment.NN_config()['model'])
    stitch_model = model_class(stitch_data_config, stitch_experiment.NN_config(), stitch_experiment.NN_config()['loss'])
    if 'device_ids' in stitch_experiment.NN_config():  # model from multi-gpu training case
        stitch_model = nn.DataParallel(stitch_model, device_ids=['cuda:0'])

    stitch_model.load_state_dict(stitch_experiment.load_best_model()['model_state_dict'])
    # stitch_model = stitch_model.to(device=device)
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

        pattern = NNSewingPattern(view_ids=False)
        pattern.name = name
        try:
            pattern.pattern_from_tensors(
                prediction['outlines'], prediction['rotations'], prediction['translations'], 
                padded=True)   
            pattern.stitches_from_pair_classifier(stitch_model, stitch_data_config['standardize'])
            pattern.serialize(save_to / 'stitched', to_subfolder=True)

        except (RuntimeError, InvalidPatternDefError, TypeError) as e:
            print(traceback.format_exc())
            print(e)
            print('Saving predictions::Skipping pattern {}'.format(name))
            pass
