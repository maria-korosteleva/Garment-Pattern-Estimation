from pathlib import Path
import customconfig
import wandb

system_props = customconfig.Properties('./system.json')

to_upload = {
    # 'dress_sleeveless_2550': 'dress_sleeveless',
    # 'jumpsuit_sleeveless_2000': 'jumpsuit_sleeveless',
    # 'skirt_8_panels_1000': 'skirt_8_panels',
    # 'wb_pants_straight_1500': 'wb_pants_straight',
    # 'skirt_2_panels_1200': 'skirt_2_panels',
    # 'jacket_2200': 'jacket',
    # 'tee_sleeveless_1800': 'tee_sleeveless',
    # 'wb_dress_sleeveless_2600': 'wb_dress_sleeveless',
    'jacket_hood_2700': 'jacket_hood',
    'pants_straight_sides_1000': 'pants_straight_sides'
    # 'tee_2300': 'tee',
    # 'skirt_4_panels_1600': 'skirt_4_panels'
    
    # 'test_150_jacket_hood_sleeveless_210331-11-16-33': 'jacket_hood_sleeveless-test', 
    # 'test_150_skirt_waistband_210331-16-05-37': 'skirt_waistband-test',
    # 'test_150_jacket_sleeveless_210331-15-54-26': 'jacket_sleeveless-test',
    # 'test_150_dress_210401-17-57-12': 'dress-test',
    # 'test_150_jumpsuit_210401-16-28-21': 'jumpsuit-test',
    # 'test_150_wb_jumpsuit_sleeveless_210404-11-27-30': 'wb_jumpsuit_sleeveless-test',
    # 'test_150_tee_hood_210401-15-25-29': 'tee_hood-test' 
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



