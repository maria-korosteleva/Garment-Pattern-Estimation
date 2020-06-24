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
    run_name='my_garments_data', 
    run_id=None, 
    no_sync=False) 

# train
dataset = data.GarmentParamsDataset(Path(system_info['output']) / dataset_folder)
trainer = Trainer(experiment, dataset, valid_percent=15, test_percent=10)
dataset_wrapper = trainer.datawraper
# model
trainer.init_randomizer()
model = nets.GarmentParamsMLP(dataset.feature_size, dataset.ground_truth_size)
# fit
trainer.fit(model)

# --------------- Final evaluation --------------
final_metrics = metrics.eval_metrics(model, dataset_wrapper, 'test')
print ('Test metrics: {}'.format(final_metrics))

experiment.add_statistic('test', final_metrics)  # TODO doesn't work for unfinished runs??? 3e2awx85

# save prediction for validation to file
dataset_wrapper.predict(model, 'test')
