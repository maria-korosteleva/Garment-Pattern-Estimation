from pathlib import Path
import wandb
import customconfig

system_props = customconfig.Properties('./system.json')

# name:version in wandb : name to save to
to_download = {
    # 'tee:latest': 'tee_2300',
    # 'tee_sleeveless:latest': 'tee_sleeveless_1800',
    # 'pants_straight_sides:latest': 'pants_straight_sides_1000'
    # 'wb_dress_sleeveless:latest': 'wb_dress_sleeveless_2600',
    # 'jacket_hood:latest': 'jacket_hood_2700',
    'jacket:latest': 'jacket_2200',
    # 'skirt_4_panels:latest': 'skirt_4_panels_1600',
    # 'skirt_2_panels:latest': 'skirt_2_panels_1200',
    # 'skirt_8_panels:latest': 'skirt_8_panels_1000',
    # 'wb_pants_straight:latest': 'wb_pants_straight_1500'

}

api = wandb.Api({'project': 'Garments-Reconstruction'})
for key in to_download:
    artifact = api.artifact(name=key)
    filepath = artifact.download(Path(system_props['datasets_path']) / to_download[key])
    # filepath = artifact.download(Path(system_props['output']) / to_download[key])
