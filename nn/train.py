from pathlib import Path

# My modules
import customconfig, data, nets, metrics
from trainer import Trainer
from experiment import WandbRunWrappper

# init
dataset_folder = 'data_1000_skirt_4_panels_200616-14-14-40'

system_info = customconfig.Properties('./system.json')
experiment = WandbRunWrappper(
    system_info['wandb_username'],
    project_name='Test-Garments-Reconstruction', 
    run_name='artifacts', 
    run_id='fr3rlv3c', 
    no_sync=False) 

# train
# dataset = data.GarmentParamsDataset(Path(system_info['output']) / dataset_folder, {'mesh_samples': 2000})
dataset = data.Garment3DParamsDataset(Path(system_info['output']) / dataset_folder, {'mesh_samples': 2000})
# dataset = data.ParametrizedShirtDataSet(r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest', {'num_verts': 'all'})
trainer = Trainer(experiment, dataset, 
                  valid_percent=10, test_percent=10, split_seed=10,
                  with_visualization=True)  # only turn on on custom garment data
dataset_wrapper = trainer.datawraper
# model
trainer.init_randomizer()
# model = nets.GarmentParamsMLP(dataset.config['feature_size'], dataset.config['ground_truth_size'])
model = nets.GarmentParamsPoint(dataset.config['ground_truth_size'], {'r1': 10, 'r2': 40})
# model = nets.ShirtfeaturesMLP(dataset.config['feature_size'], dataset.config['ground_truth_size'])
if hasattr(model, 'config'):
    trainer.update_config(NN=model.config)  # save NN configuration

# fit
trainer.fit(model)

# --------------- Final evaluation --------------
final_metrics = metrics.eval_metrics(model, dataset_wrapper, 'test')
print ('Test metrics: {}'.format(final_metrics))
experiment.add_statistic('test', final_metrics)

# save predictions
prediction_path = dataset_wrapper.predict(model, save_to=Path(system_info['output']), sections=['validation', 'test'])
print('Predictions saved to {}'.format(prediction_path))

# reflect predictions info in expetiment
experiment.add_statistic('predictions_folder', prediction_path.name)
experiment.add_artifact(prediction_path, dataset_wrapper.dataset.name, 'result')



