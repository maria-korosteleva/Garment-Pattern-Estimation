"""Predicting a 2D pattern for the given 3D models of garments -- not necessarily from the garment datasets of this project"""

import argparse
from datetime import datetime
import igl
import numpy as np
from pathlib import Path
import shutil
import torch

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys,inspect

from nn.pattern_converter import NNSewingPattern
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# My modules
import customconfig, nets, data
from experiment import WandbRunWrappper
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
    shape_experiment = WandbRunWrappper(
        system_info['wandb_username'],
        project_name='Garments-Reconstruction', 
        run_name='multi-all-fin', 
        run_id='216nexgv')  # finished experiment
    stitch_experiment = WandbRunWrappper(
        system_info['wandb_username'],
        project_name='Garments-Reconstruction', 
        run_name='multi-all-fin', 
        run_id='216nexgv')  # finished experiment
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
            raise ValueError('Input point cloud has {} points while {} are expected'.format(points.shape[0], data_config['mesh_samples']))

        if 'standardize' in data_config:
            points = (points - data_config['standardize']['f_shift']) / data_config['standardize']['f_scale']
        points_list.append(torch.tensor(points).float())

    # ----- Model (Pattern Shape) architecture -----
    model_class = getattr(nets, shape_experiment.NN_config()['model'])
    model = model_class(data_config, shape_experiment.NN_config(), shape_experiment.NN_config()['loss'])
    model.load_state_dict(shape_experiment.load_best_model()['model_state_dict'])
    model = model.to(device=device)
    model.eval()

    # -------- Predict Shape ---------
    with torch.no_grad():
        points_batch = torch.stack(points_list).to(device)
        predictions = model(points_batch)

    # ---- save shapes ----
    saving_path = save_to / 'shape'
    saving_path.mkdir(parents=True, exists_ok=True)

    names = [VisPattern.name_from_path(elem) for elem in sample_paths]
    data.save_garments_prediction(predictions, saving_path, data_config, names)


    # ========== Stitch prediction =========

    # ----- Model (Stitch Prediction) ------
    _, _, data_config = stitch_experiment.data_info()  # need to get data stats
    model_class = getattr(nets, stitch_experiment.NN_config()['model'])
    stitch_model = model_class(data_config, stitch_experiment.NN_config(), stitch_experiment.NN_config()['loss'])
    stitch_model.load_state_dict(stitch_experiment.load_best_model()['model_state_dict'])
    stitch_model = stitch_model.to(device=device)
    stitch_model.eval()

    # ----- predict & save stitches ------
    saving_path = save_to / 'stitched'
    saving_path.mkdir(parents=True, exists_ok=True)
    for idx, name in enumerate(names):
        # "unbatch" dictionary
        prediction = {}
        for key in predictions:
            prediction[key] = predictions[key][idx]
        pattern = NNSewingPattern(view_ids=False)
        pattern.name = name
        try:
            pattern.pattern_from_tensors(
                prediction['outlines'], prediction['rotations'], prediction['translations'], 
                padded=True)   
            pattern.stitches_from_pair_classifier(stitch_model, data_config['standardize'])
            pattern.serialize(save_to / 'stitched', to_subfolder=True)

        except (RuntimeError, InvalidPatternDefError, TypeError) as e:
            print(e)
            print('Saving predictions::Skipping pattern {}'.format(name))
            pass
