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
currentdir = os.path.dirname(os.path.realpath(__file__) )
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

# My modules
import customconfig, nets
from experiment import WandbRunWrappper
from pattern.wrappers import VisPattern
from data import GarmentBaseDataset

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
        default='per_mesh_pred')

    args = parser.parse_args()
    print(args)

    # turn arguments into the list of obj files
    paths_list = []
    if args.file is None and args.directory is None: 
        # default value if no arguments provided
        paths_list.append(Path('D:/Data/my garments/data_1000_tee_200527-14-50-42_regen_200612-16-56-43/tee_1NOVZ1DB7L/tee_1NOVZ1DB7L_sim.obj'))
    else:
        if args.file is not None:
            paths_list.append(Path(args.file))
        if args.directory is not None:
            directory = Path(args.directory)
            for elem in directory.glob('*'):
                if elem.is_file() and '.obj' in elem:
                    paths_list.append(elem)

    saving_path = Path(system_info['output']) / (args.save_tag + '_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
    saving_path.mkdir(parents=True)

    return paths_list, saving_path


if __name__ == "__main__":
    
    system_info = customconfig.Properties('./system.json')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mesh_paths, save_to = get_meshes_from_args()

    # --------------- Experiment to evaluate on ---------
    experiment = WandbRunWrappper(system_info['wandb_username'],
        project_name='Test-Garments-Reconstruction', 
        run_name='Pattern3D-data-transforms', 
        run_id='cgkk8eb7') 
    if not experiment.is_finished():
        print('Warning::Evaluating unfinished experiment')

    # data stats from training 
    _, _, data_config = experiment.data_info()  # need to get data stats

    # ----- Model architecture -----
    # model = nets.GarmentPanelsAE(dataset.config['element_size'], dataset.config['feature_size'], experiment.NN_config())
    # NOTE this is an old model. This script needs updating!!!
    model = nets.GarmentPattern3D(data_config, experiment.NN_config())
    
    # model.load_state_dict(experiment.load_final_model()['model_state_dict'])
    # model.load_state_dict(experiment.load_checkpoint_file()['model_state_dict'])
    model.load_state_dict(experiment.load_best_model()['model_state_dict'])
    model = model.to(device=device)

    # ------ prepare input data & construct batch -------
    points_list = []
    for mesh in mesh_paths:
        verts, faces = igl.read_triangle_mesh(str(mesh))
        points = GarmentBaseDataset.sample_mesh_points(data_config['mesh_samples'], verts, faces)
        if 'standardize' in data_config:
            points = (points - data_config['standardize']['f_shift']) / data_config['standardize']['f_scale']
        points_list.append(torch.Tensor(points))

    # -------- Predict ---------
    with torch.no_grad():
        points_batch = torch.stack(points_list).to(device)
        preds = model(points_batch).cpu().numpy()

    # ---- save ----
    for pred, mesh in zip(preds, mesh_paths):
        if 'standardize' in data_config:
            pred = pred * data_config['standardize']['scale'] + data_config['standardize']['shift']

        pattern = VisPattern(view_ids=False)
        pattern.name = VisPattern.name_from_path(mesh)
        pattern.pattern_from_tensor(pred, padded=True)   

        log_dir = pattern.serialize(save_to)
        shutil.copy2(str(mesh), str(log_dir))