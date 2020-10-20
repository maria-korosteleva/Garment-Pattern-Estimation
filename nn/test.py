"""Evaluate a model on the data"""

from pathlib import Path
import torch

# My modules
import customconfig
import data
import metrics
import nets
from trainer import Trainer
from experiment import WandbRunWrappper

# --------------- from experimnet ---------
system_info = customconfig.Properties('./system.json')
experiment = WandbRunWrappper(
    system_info['wandb_username'],
    project_name='Test-Garments-Reconstruction', 
    run_name='Pattern3D-data-transforms', 
    run_id='cgkk8eb7')  # finished experiment

if not experiment.is_finished():
    print('Warning::Evaluating unfinished experiment')

# -------- data -------
split, batch_size, data_config = experiment.data_info()  # note that run is not initialized -- we use info from finished run

datapath = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
# data_config.update({'num_verts': 500})
# dataset = data.ParametrizedShirtDataSet(datapath, data_config)
# dataset_folder = 'data_1000_skirt_4_panels_200616-14-14-40'
dataset_folder = 'data_1000_tee_200527-14-50-42_regen_200612-16-56-43'
# dataset = data.GarmentParamsDataset(Path(system_info['datasets_path']) / dataset_folder, data_config)
# dataset = data.Garment3DParamsDataset(Path(system_info['datasets_path']) / dataset_folder, data_config, gt_caching=True, feature_caching=True)
# dataset = data.GarmentPanelDataset(Path(system_info['datasets_path']) / data_config['name'], data_config)
dataset = data.Garment3DPatternDataset(
    Path(system_info['datasets_path']) / dataset_folder, 
    data_config, 
    gt_caching=True, feature_caching=True)

print(dataset.config)
print('Batch: {}, Split: {}'.format(batch_size, split))

datawrapper = data.DatasetWrapper(dataset, known_split=split, batch_size=batch_size)

# ----- Model architecture -----
# model = nets.ShirtfeaturesMLP(dataset.config['feature_size'], dataset.config['ground_truth_size'])
# model = nets.GarmentParamsMLP(dataset.config['feature_size'], dataset.config['ground_truth_size'])
# model = nets.GarmentParamsPoint(dataset.config['ground_truth_size'], experiment.NN_config())
# model = nets.GarmentPanelsAE(dataset.config['element_size'], dataset.config['feature_size'], experiment.NN_config())
model = nets.GarmentPattern3D(
    dataset.config['element_size'], dataset.config['panel_len'], dataset.config['ground_truth_size'], dataset.config['standardize'],
    experiment.NN_config()
)

# model_state = torch.load('./wandb/artifacts/2lunqzha/checkpoint_227.pth')['model_state_dict']  # debug
# model.load_state_dict(model_state)

# model.load_state_dict(experiment.load_final_model())
# model.load_state_dict(experiment.load_checkpoint_file()['model_state_dict'])
model.load_state_dict(experiment.load_best_model()['model_state_dict'])

# ------- Evaluate --------
valid_loss = metrics.eval_metrics(model, datawrapper, 'validation', loop_loss=True)
print('Validation metrics: {}'.format(valid_loss))
test_metrics = metrics.eval_metrics(model, datawrapper, 'test', loop_loss=True)
print('Test metrics: {}'.format(test_metrics))

# print(dataset[276]['features'])  # first element of validation set

experiment.add_statistic('valid_metrics', valid_loss)
experiment.add_statistic('test_metrics', test_metrics)

# -------- Predict ---------
# save prediction for validation to file
prediction_path = datawrapper.predict(model, save_to=Path(system_info['output']), sections=['validation', 'test'])
print('Saved to {}'.format(prediction_path))
