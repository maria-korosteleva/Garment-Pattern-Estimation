from pathlib import Path
import wandb
import customconfig

system_props = customconfig.Properties('./system.json')

# name:version in wandb : name to save to
to_download = {
    # 'tee:latest': 'tee_2300',
    'tee_sleeveless:latest': 'tee_sleeveless_1800',
    'pants_straight_sides:latest': 'pants_straight_sides_1000'
}

api = wandb.Api({'project': 'Garments-Reconstruction'})
for key in to_download:
    artifact = api.artifact(name=key)
    filepath = artifact.download(Path(system_props['datasets_path']) / to_download[key])
    # filepath = artifact.download(Path(system_props['output']) / to_download[key])
