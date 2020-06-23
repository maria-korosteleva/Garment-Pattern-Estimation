from pathlib import Path

# My modules
import customconfig, data, nets, metrics
from trainer import Trainer
from experiment import WandbRunWrappper

# init
datapath = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
system_info = customconfig.Properties('./system.json')
experiment = WandbRunWrappper(
    system_info['wandb_username'],
    project_name='Test-Garments-Reconstruction', 
    run_name='wb_wrapper', 
    run_id=None, 
    no_sync=False) 

# train
trainer = Trainer(experiment, data.ParametrizedShirtDataSet(Path(datapath)), valid_percent=10)
shirts_wrapper = trainer.datawraper
# model
trainer.init_randomizer()
model = nets.ShirtfeaturesMLP()
# fit
trainer.fit(model)

# --------------- Final tests on validation set --------------
experiment.stop()  # Test on finished runs
valid_loss = metrics.eval_metrics(model, shirts_wrapper, 'validation')
print ('Validation loss: {}'.format(valid_loss))

experiment.add_statistic('valid_metrics', valid_loss)

# save prediction for validation to file
shirts_wrapper.predict(model, 'validation')