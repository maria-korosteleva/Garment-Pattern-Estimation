from pathlib import Path
import customconfig
import wandb

system_props = customconfig.Properties('./system.json')

to_upload = {
    'jacket_hood_2700': 'jacket_hood',
    'pants_straight_sides_1000': 'pants_straight_sides'
    # 'tee_2300': 'tee',
}

for dataset, art_name in to_upload.items():
    wandb.init(project='Garments-Reconstruction', job_type='dataset')

    artifact = wandb.Artifact(art_name, type='dataset', description= dataset + ' + props cleanup')
    # Add a file to the artifact's contents
    datapath = Path(system_props['datasets_path']) / dataset
    artifact.add_dir(str(datapath))
    # Save the artifact version to W&B and mark it as the output of this run
    wandb.run.log_artifact(artifact, aliases=['latest', 'nips21'])

    wandb.finish()  # sync all data before moving to next dataset



