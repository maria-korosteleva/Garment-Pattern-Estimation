"""Interpolate sewing patterns from the two input 3D garments"""

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
import customconfig, nets, data
from experiment import WandbRunWrappper
from pattern.wrappers import VisPattern
from data import GarmentBaseDataset


if __name__ == "__main__":
    
    system_info = customconfig.Properties('./system.json')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #save_to = Path(system_info['output']) / ('interpolate' + '_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
    save_to = Path('D:\MyDocs\GigaKorea\SIGGRAPH2021 submission materials\Interpolation') / ('pants_interpolate' + '_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
    save_to.mkdir(parents=True)

    # exactly 2
    mesh_paths = [
        # Path(system_info['datasets_path']) / 'data_1000_tee_200527-14-50-42_regen_200612-16-56-43' / 'tee_6ZX9WWODMX' / 'tee_6ZX9WWODMX_sim.obj',  # long and wide
        # Path(system_info['datasets_path']) / 'data_1000_tee_200527-14-50-42_regen_200612-16-56-43' / 'tee_4U1YS26K29' / 'tee_4U1YS26K29_sim.obj',  # short, long sleeves
        # Path(system_info['datasets_path']) / 'data_1000_pants_straight_sides_210105-10-49-02' / 'pants_straight_sides_1VDATY2TSF' / 'pants_straight_sides_1VDATY2TSF_sim.obj',
        Path(system_info['datasets_path']) / 'data_1000_pants_straight_sides_210105-10-49-02' / 'pants_straight_sides_SAD7HA2LGE' / 'pants_straight_sides_SAD7HA2LGE_sim.obj',
        Path(system_info['datasets_path']) / 'data_1000_pants_straight_sides_210105-10-49-02' / 'pants_straight_sides_SVFYS0EX50' / 'pants_straight_sides_SVFYS0EX50_sim.obj',
        # Path(system_info['datasets_path']) / 'data_1000_skirt_4_panels_200616-14-14-40' / 'skirt_4_panels_2H2PV4GECN' / 'skirt_4_panels_2H2PV4GECN_sim.obj',  # short & slim skirt
        # Path(system_info['datasets_path']) / 'data_1000_skirt_4_panels_200616-14-14-40' / 'skirt_4_panels_WV3Z327KQ8' / 'skirt_4_panels_WV3Z327KQ8_sim.obj',  # short & slim skirt
    ]
    num_in_between = 5

    # --------------- Experiment to evaluate on ---------
    experiment = WandbRunWrappper(system_info['wandb_username'],
        project_name='Garments-Reconstruction', 
        run_name='multi-all-fin', 
        run_id='216nexgv')  # finished experiment
    if not experiment.is_finished():
        print('Warning::Evaluating unfinished experiment')

    # data stats from training 
    _, _, data_config = experiment.data_info()  # need to get data stats

    # ----- Model architecture -----
    model = nets.GarmentFullPattern3D(data_config, experiment.NN_config())
    model.load_state_dict(experiment.load_best_model()['model_state_dict'])
    model = model.to(device=device)
    model.eval()

    # ------ prepare input data & construct batch -------
    points_list = data.sample_points_from_meshes(mesh_paths, data_config)

    # -------- Interpolation ---------
    # Encode
    with torch.no_grad():
        points_batch = torch.stack(points_list).to(device)
        pred_encodings = model.forward_encode(points_batch)

    # Interpolate
    encodings = []
    t = 0
    encodings.append(pred_encodings[0])
    for i in range(num_in_between + 1):
        t += 1. / (num_in_between + 1)
        encodings.append(
            (1 - t) * pred_encodings[0] + t * pred_encodings[1]
        )
    encodings = torch.stack(encodings).to(device)
    # encodings = pred_encodings

    # print(encodings)

    # decode
    with torch.no_grad():
        preds = model.forward_decode(encodings)

    # ---- save ----
    names = ['t_{:.2f}'.format(i / (num_in_between + 1)) for i in range(num_in_between + 2)]

    data.save_garments_prediction(preds, save_to, data_config, names)
