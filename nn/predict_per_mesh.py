"""Predicting a 2D pattern for the given 3D models of garments -- not necessarily from the garment datasets of this project"""

from pathlib import Path
import torch
import igl
import numpy as np

# My modules
import customconfig, data, metrics, nets
from trainer import Trainer
from experiment import WandbRunWrappper
from pattern.wrappers import VisPattern


# --------------- Experiment to evaluate on ---------
system_info = customconfig.Properties('./system.json')
experiment = WandbRunWrappper(system_info['wandb_username'],
    project_name='Test-Garments-Reconstruction', 
    run_name='Pattern3D-data-transforms', 
    run_id='cgkk8eb7')  # finished experiment

if not experiment.is_finished():
    print('Warning::Evaluating unfinished experiment')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# -------- data stats from training -------
_, _, data_config = experiment.data_info()  # need to get data stats

# ----- Model architecture -----
# model = nets.GarmentPanelsAE(dataset.config['element_size'], dataset.config['feature_size'], experiment.NN_config())
model = nets.GarmentPattern3D(
    data_config['element_size'], data_config['panel_len'], data_config['ground_truth_size'], data_config['standardize'],
    experiment.NN_config()
)
# model.load_state_dict(experiment.load_final_model())
# model.load_state_dict(experiment.load_checkpoint_file()['model_state_dict'])
model.load_state_dict(experiment.load_best_model()['model_state_dict'])
model = model.to(device=device)

# ------ prepare input data -------
# TODO get from paramters
# TODO allow to specify folder
mesh_path = Path('D:/Data/my garments/data_1000_tee_200527-14-50-42_regen_200612-16-56-43/tee_1NOVZ1DB7L/tee_1NOVZ1DB7L_sim.obj')
verts, faces = igl.read_triangle_mesh(str(mesh_path))

# sample 
# TODO remove duplicate code
barycentric_samples, face_ids = igl.random_points_on_mesh(data_config['mesh_samples'], verts, faces)
face_ids[face_ids >= len(faces)] = len(faces) - 1  # workaround for https://github.com/libigl/libigl/issues/1531

# convert to normal coordinates
points = np.empty(barycentric_samples.shape)
for i in range(len(face_ids)):
    face = faces[face_ids[i]]
    barycentric_coords = barycentric_samples[i]
    face_verts = verts[face]
    points[i] = np.dot(barycentric_coords, face_verts)

# standardize
if 'standardize' in data_config:
    points = (points - data_config['standardize']['f_mean']) / data_config['standardize']['f_std']

# torch batch
points = torch.Tensor(points)
points = points.unsqueeze(0).to(device)


# -------- Predict ---------
with torch.no_grad():
    preds = model(points).cpu().numpy()

# ---- save ----
for pred in preds:
    print(preds)
    if 'standardize' in data_config:
        pred = pred * data_config['standardize']['std'] + data_config['standardize']['mean']

    pattern = VisPattern(view_ids=False)
    pattern.name = mesh_path.stem

    try: 
        pattern.pattern_from_tensor(pred, padded=True)   
    except RuntimeError as e:
        print('Garment3DPatternDataset::Warning::{}: {}'.format(mesh_path.stem, e))
        pass

    pattern.serialize('./wandb', tag='pred_')