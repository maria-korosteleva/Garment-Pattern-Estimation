"""Evaluate a model on the data"""

from pathlib import Path

# My modules
import customconfig, data, metrics, nets
from trainer import Trainer
from experiment import WandbRunWrappper

# --------------- from experimnet ---------
system_info = customconfig.Properties('./system.json')
experiment = WandbRunWrappper(
    system_info['wandb_username'],
    project_name='Test-Garments-Reconstruction', 
    run_name='panelAE-tee-norm', 
    run_id='3cq6lnyt')  # finished experiment

if not experiment.is_finished():
    print('Warning::Evaluating unfinished experiment')

# -------- data -------
split, batch_size, data_config = experiment.data_info()  # note that run is not initialized -- we use info from finished run

datapath = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
# data_config.update({'num_verts': 500})
# dataset = data.ParametrizedShirtDataSet(datapath, data_config)
# dataset_folder = 'data_1000_skirt_4_panels_200616-14-14-40'
dataset_folder = 'data_1000_tee_200527-14-50-42_regen_200612-16-56-43'
# dataset = data.GarmentParamsDataset(Path(system_info['output']) / dataset_folder, data_config)
# dataset = data.Garment3DParamsDataset(Path(system_info['output']) / dataset_folder, data_config)
dataset = data.GarmentPanelDataset(Path(system_info['output']) / data_config['name'], data_config)

print(dataset.config)

datawrapper = data.DatasetWrapper(dataset, known_split=split, batch_size=batch_size)

# ----- Model architecture -----
# model = nets.ShirtfeaturesMLP(dataset.config['feature_size'], dataset.config['ground_truth_size'])
# model = nets.GarmentParamsMLP(dataset.config['feature_size'], dataset.config['ground_truth_size'])
# model = nets.GarmentParamsPoint(dataset.config['ground_truth_size'], experiment.NN_config())
model = nets.GarmentPanelsAE(dataset.config['element_size'], dataset.config['feature_size'], experiment.NN_config())
model.load_state_dict(experiment.load_final_model())
# model.load_state_dict(experiment.load_checkpoint_file()['model_state_dict'])

# ------- Evaluate --------
# valid_loss = metrics.eval_metrics(model, datawrapper, 'validation')
# print ('Validation metrics: {}'.format(valid_loss))
# test_metrics = metrics.eval_metrics(model, datawrapper, 'test')
# print ('Test metrics: {}'.format(test_metrics))
# experiment.add_statistic('valid_metrics', valid_loss)

# -------- Predict ---------
# save prediction for validation to file
prediction_path = datawrapper.predict(model, save_to=Path(system_info['output']), sections=['validation', 'test'])

print('Saved to {}'.format(prediction_path))