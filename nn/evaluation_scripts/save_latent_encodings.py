from pathlib import Path
import torch
import pickle
import numpy as np

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys,inspect
currentdir = os.path.dirname(os.path.realpath(__file__) )
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

# My modules
import customconfig
import data
import nets
from trainer import Trainer
from experiment import WandbRunWrappper

# --------------- from experimnet ---------
system_info = customconfig.Properties('./system.json')
experiment = WandbRunWrappper(
    system_info['wandb_username'],
    project_name='Garments-Reconstruction', 
    run_name='multi-all-fin', 
    run_id='216nexgv')  # finished experiment

if not experiment.is_finished():
    print('Warning::Evaluating unfinished experiment')

# -------- data -------
# data_config also contains the names of datasets to use
split, batch_size, data_config = experiment.data_info()  # note that run is not initialized -- we use info from finished run
data_config.update({'obj_filetag': 'sim'})  # scan imitation stats

dataset = data.Garment3DPatternFullDataset(
    system_info['datasets_path'], data_config, gt_caching=True, feature_caching=True)
datawrapper = data.DatasetWrapper(dataset, known_split=split, batch_size=batch_size)
test_loader = datawrapper.get_loader('test')

# ----- Model -------
model = nets.GarmentFullPattern3D(dataset.config, experiment.NN_config())
model.load_state_dict(experiment.load_best_model()['model_state_dict'])

# ----- Analysis -----

# get all encodings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
all_encodings = []
classes = []
with torch.no_grad():
    for batch in test_loader:
        features = batch['features'].to(device)
        garment_encodings = model.forward_encode(features)

        all_encodings.append(garment_encodings)
        classes += batch['data_folder']

all_encodings = torch.cat(all_encodings).cpu().numpy()

np.save('tmp_enc.npy', all_encodings)
with open('tmp_data_folders.pkl', 'wb') as fp:
    pickle.dump(classes, fp)