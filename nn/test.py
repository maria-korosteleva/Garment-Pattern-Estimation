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
    project_name='Garments-Reconstruction', 
    run_name='panel-origin-fix', 
    run_id='1svo6wuc')  # finished experiment

if not experiment.is_finished():
    print('Warning::Evaluating unfinished experiment')

# -------- data -------
# data_config also contains the names of datasets to use
split, batch_size, data_config = experiment.data_info()  # note that run is not initialized -- we use info from finished run

datapath = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
# data_config.update({'num_verts': 500})
# dataset = data.ParametrizedShirtDataSet(datapath, data_config)
# dataset = data.GarmentParamsDataset(system_info['datasets_path'], data_config)
# dataset = data.Garment3DParamsDataset(system_info['datasets_path'], data_config, gt_caching=True, feature_caching=True)
# dataset = data.GarmentPanelDataset(system_info['datasets_path'], data_config)
dataset = data.Garment3DPatternFullDataset(
    system_info['datasets_path'], data_config, gt_caching=True, feature_caching=True)

print(dataset.config)
print('Batch: {}, Split: {}'.format(batch_size, split))

datawrapper = data.DatasetWrapper(dataset, known_split=split, batch_size=batch_size)

# ----- Model architecture -----
# model = nets.ShirtfeaturesMLP(dataset.config['feature_size'], dataset.config['ground_truth_size'])
# model = nets.GarmentParamsMLP(dataset.config['feature_size'], dataset.config['ground_truth_size'])
# model = nets.GarmentParamsPoint(dataset.config['ground_truth_size'], experiment.NN_config())
# model = nets.GarmentPanelsAE(dataset.config['element_size'], dataset.config['feature_size'], experiment.NN_config())
model = nets.GarmentFullPattern3D(dataset.config, experiment.NN_config())

model.load_state_dict(experiment.load_best_model()['model_state_dict'])

# ------- Evaluate --------
valid_loss = metrics.eval_metrics(model, datawrapper, 'validation')
print('Validation metrics: {}'.format(valid_loss))
valid_breakdown = metrics.eval_metrics(model, datawrapper, 'valid_per_data_folder')
print('Validation metrics per dataset: {}'.format(valid_breakdown))

test_metrics = metrics.eval_metrics(model, datawrapper, 'test')
print('Test metrics: {}'.format(test_metrics))
test_breakdown = metrics.eval_metrics(model, datawrapper, 'test_per_data_folder')
print('Test metrics per dataset: {}'.format(test_breakdown))

# print(dataset[276]['features'])  # first element of validation set

# experiment.add_statistic('valid_on_best', valid_loss)
# experiment.add_statistic('valid_best_breakdown', valid_breakdown)
# experiment.add_statistic('test_on_best', test_metrics)
# experiment.add_statistic('test_best_breakdown', test_breakdown)

# -------- Predict ---------
# save prediction for validation to file
prediction_path = datawrapper.predict(model, save_to=Path(system_info['output']), sections=['validation', 'test'])
print('Saved to {}'.format(prediction_path))
# reflect predictions info in expetiment
# experiment.add_statistic('predictions_folder', prediction_path.name)
# experiment.add_artifact('D:/GK-Pattern-Data-Gen/nn_pred_data_1000_tee_200527-14-50-42_regen_200612-16-56-43201113-17-27-49', datawrapper.dataset.name, 'result')