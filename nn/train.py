from pathlib import Path

# My modules
import customconfig, data, nets, metrics
from trainer import Trainer
from experiment import WandbRunWrappper

# init
# dataset_folder = 'data_1000_skirt_4_panels_200616-14-14-40'
dataset_folder = 'data_1000_tee_200527-14-50-42_regen_200612-16-56-43'

system_info = customconfig.Properties('./system.json')
# MAIN experiment object
experiment = WandbRunWrappper(
    system_info['wandb_username'],
    project_name='Garments-Reconstruction', 
    run_name='Pattern3D-capacity',   # Pattern3D-tee-in-std
    run_id=None, 
    no_sync=False) 

# get data stats from older runs to save runtime
old_experiment = WandbRunWrappper(system_info['wandb_username'],
    project_name='Garments-Reconstruction', 
    run_name='Pattern3D-capacity',
    run_id='2hj3sdv0'
)
# NOTE data stats are ONLY correct for a specific data split, so these two need to go together
old_split, _, data_config = old_experiment.data_info()

# train
# dataset = data.ParametrizedShirtDataSet(r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest', {'num_verts': 'all'})
# dataset = data.GarmentParamsDataset(Path(system_info['datasets_path']) / dataset_folder, {'mesh_samples': 2000})
# dataset = data.Garment3DParamsDataset(Path(system_info['datasets_path']) / dataset_folder, {'mesh_samples': 2000}, gt_caching=True, feature_caching=True)
# dataset = data.GarmentPanelDataset(
#     Path(system_info['datasets_path']) / dataset_folder, 
#     {'panel_name': 'front'}, 
#     gt_caching=True, feature_caching=True)
# dataset = data.Garment2DPatternDataset(Path(system_info['datasets_path']) / dataset_folder, gt_caching=True, feature_caching=True)
# NOTE this dataset involves point sampling SO data stats from previous runs might not be correct,
# Especially if we change the number of samples
# TO speed up training, we assume that previous estimations are close to the truth (as the data amount is sufficient for good estimation)
data_config.update(mesh_samples=4000)
dataset = data.Garment3DPatternDataset(
    Path(system_info['datasets_path']) / dataset_folder, 
    data_config,   # uses data stats from previous run
    gt_caching=True, feature_caching=True)

trainer = Trainer(experiment, dataset, 
                  valid_percent=old_split['valid_percent'],   
                  test_percent=old_split['test_percent'], 
                  split_seed=old_split['random_seed'],   # valid_percent=10, test_percent=10, split_seed=10,
                  with_norm=True,
                  with_visualization=False)  # only turn on on custom garment data
dataset_wrapper = trainer.datawraper
# model
trainer.init_randomizer(100)
# model = nets.ShirtfeaturesMLP(dataset.config['feature_size'], dataset.config['ground_truth_size'])
# model = nets.GarmentParamsMLP(dataset.config['feature_size'], dataset.config['ground_truth_size'])
# model = nets.GarmentParamsPoint(dataset.config['ground_truth_size'], {'r1': 10, 'r2': 40})
# model = nets.GarmentPanelsAE(
#     dataset.config['element_size'], dataset.config['feature_size'], dataset.config['standardize'],
#     {'hidden_dim_enc': 25, 'hidden_dim_dec': 25, 'n_layers': 3, 'loop_loss_weight': 0.1, 'dropout': 0})
# model = nets.GarmentPatternAE(
#     dataset.config['element_size'], dataset.config['panel_len'], dataset.config['standardize'],
#     {
#         'panel_encoding_size': 25, 'panel_n_layers': 3, 
#         'pattern_encoding_size': 40, 'pattern_n_layers': 3, 
#         'loop_loss_weight': 0.1, 'dropout': 0
#     }
# ) 
model = nets.GarmentPattern3DPoint(
    dataset.config['element_size'], dataset.config['panel_len'], dataset.config['ground_truth_size'], dataset.config['standardize'],
    {
        'r1': 1, 'r2': 4,
        'panel_encoding_size': 35, 'panel_n_layers': 3, 
        'pattern_encoding_size': 100, 'pattern_n_layers': 3, 
        'loop_loss_weight': 0.1, 'dropout': 0
    }
)

if hasattr(model, 'config'):
    trainer.update_config(NN=model.config)  # save NN configuration

# fit
trainer.fit(model)

# --------------- Final evaluation --------------
# save predictions
prediction_path = dataset_wrapper.predict(model, save_to=Path(system_info['output']), sections=['validation', 'test'])
print('Predictions saved to {}'.format(prediction_path))

final_metrics = metrics.eval_metrics(model, dataset_wrapper, 'test', loop_loss=True)
print ('Test metrics: {}'.format(final_metrics))
experiment.add_statistic('test', final_metrics)

# reflect predictions info in expetiment
experiment.add_statistic('predictions_folder', prediction_path.name)
experiment.add_artifact(prediction_path, dataset_wrapper.dataset.name, 'result')