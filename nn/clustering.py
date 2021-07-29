"""
    Cluster analysis on the dataset of garment patterns to 
        * identify similar types of panels across pattern topologies
        * Define panel ordering (within patterns) that exploit these similarities and differences 
"""
from pathlib import Path
# import numpy as np
import torch

# My Modules
import customconfig
import data
from experiment import WandbRunWrappper
import gap
from trainer import Trainer

# TODO cuda processing


def run(datawrapper, config={}):
    """Perform the task of clustering and data order update"""

    features = gather_features(datawrapper, config)

    k_optimal, train_labels, cluster_centers = cluster_analysis(features, config)

    print(cluster_centers)
    print(k_optimal, len(cluster_centers),)


def gather_features(datawrapper, config):
    """Gather features from given data loader in single batch to perform cluster analysis """

    dataset = datawrapper.dataset
    train_ids_subset = datawrapper.training.indices 

    features = []

    dim_sum = 0
    for idx in train_ids_subset:
        # get sample feature
        data_sample = dataset[idx]
        pattern_info = data_sample['ground_truth']   # TODO Refactoring -- Rename gt to something?? -- 
        num_panels = pattern_info['num_panels']
        dim_sum += num_panels  # DEBUG ing

        if config['cluster_by'] == 'translation':
            # without padded part
            features.append(pattern_info['translations'][:num_panels])
        else:
            raise NotImplemented(f'Clustering::Error::clustering by {config["cluster_by"]} is not impelemnted')

    # Gather features into single tenzor
    features = torch.cat(features, dim=0)  # cat allows tenzors to have different size in dim
    
    return features.view(-1, features.shape[-1])


def cluster_analysis(data_features, config):
    """Find clusters in input data features"""
    if config['analysis_type'] == 'gap':
        k_optimal, diff, labels, cluster_centers = gap.gaps(
            data_features,
            max_k=config['max_k'], 
            sencitivity_threshold=config['gap_cluster_threshold'], 
            logs=True  # TODO controller?
        )
    elif config['analysis_type'] == 'heuristic':
        k_optimal, diff, labels, cluster_centers = gap.optimal_clusters(
            data_features,
            max_k=config['max_k'], 
            sencitivity_threshold=config['diff_cluster_threshold'], 
            logs=True
        )
    else:
        raise NotImplemented(f'Clustering::Error::cluster analysis with {config["analysis_type"]} is not impelemnted')

    print(diff)

    return k_optimal, labels, cluster_centers


def panel_order_updates(datawrapper, cluster_centers):
    """ Re-order panels of each pattern example in the dataset 
        according to clusters they belong to """
    pass


if __name__ == "__main__":
    # np.set_printoptions(precision=4, suppress=True)  # for readability

    dataset_list = [
        # 'dress_sleeveless_2550',
        'jumpsuit_sleeveless_2000',
        # 'skirt_8_panels_1000',
        # 'wb_pants_straight_1500',
        # 'skirt_2_panels_1200',
        # 'jacket_2200',
        'tee_sleeveless_1800',
        # 'wb_dress_sleeveless_2600',
        # 'jacket_hood_2700',
        'pants_straight_sides_1000',
        'tee_2300',
        # 'skirt_4_panels_1600'
    ]
    system_info = customconfig.Properties('./system.json')
    experiment = WandbRunWrappper(
        system_info['wandb_username'], 
        project_name='Test-Garments-Reconstruction',  
        run_name='Data-clustering', 
        run_id=None, no_sync=True)   

    # Config
    split = {'valid_per_type': 150, 'test_per_type': 150, 'random_seed': 10, 'type': 'count'}   # , 'filename': './wandb/data_split.json'} 
    data_config = {
        'max_datapoints_per_type': 800,  # upper limit of how much data to grab from each type
        'max_pattern_len': 15,  # DEBUG 30 > then the total number of panel classes  
        'max_panel_len': 14,  # (jumpsuit front)
        'max_num_stitches': 24  # jumpsuit (with sleeves)
    }  
    net_seed = 10
    clustering_config = {
        'cluster_by': 'translation',  # 'translation'
        'max_k': data_config['max_pattern_len'],
        'gap_cluster_threshold': 0.0,
        'diff_cluster_threshold': 0.05,
        'analysis_type': 'gap'  # 'gap', 'heuristic'
    }

    # Get data wrapper
    data_config.update(data_folders=dataset_list)
    dataset = data.Garment2DPatternDataset(
        Path(system_info['datasets_path']), data_config, gt_caching=True, feature_caching=True)

    trainer = Trainer(experiment, dataset, split, with_norm=True, with_visualization=True)
    trainer.init_randomizer(net_seed)

    datawrapper = trainer.datawraper

    # -- Run cluster analysis --    
    run(datawrapper, clustering_config)
