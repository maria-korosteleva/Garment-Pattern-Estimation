from pathlib import Path
import argparse

# My modules
import customconfig, data, nets, metrics
from trainer import Trainer
from experiment import WandbRunWrappper

def get_values_from_args():
    """command line arguments to control the run for running wandb Sweeps!"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()
    
    # Default values from run 9j12qp24, best of sweep y1mmngej
    parser.add_argument('--mesh_samples_multiplier', '-m', help='number of samples per mesh as multiplier of 500', type=int, default=19)
    parser.add_argument('--pattern_encoding_multiplier', '-pte', help='size of pattern encoding as multiplier of 10', type=int, default=13)
    parser.add_argument('--pattern_n_layers', '-ptl', help='number of layers in pattern decoder', type=int, default=3)
    parser.add_argument('--panel_encoding_multiplier', '-pe', help='size of panel encoding as multiplier of 10', type=int, default=7)
    parser.add_argument('--panel_n_layers', '-pl', help='number of layers in panel decoder', type=int, default=4)
    parser.add_argument('--net_seed', '-ns', help='random seed for net initialization', type=float, default=916143406)

    args = parser.parse_args()
    print(args)

    data_config = {
        'mesh_samples': args.mesh_samples_multiplier * 500
    }

    nn_config = {
        'panel_encoding_size': args.panel_encoding_multiplier * 10,
        'panel_n_layers': args.panel_n_layers,
        'pattern_encoding_size': args.pattern_encoding_multiplier * 10,
        'pattern_n_layers': args.pattern_n_layers
    }


    return data_config, nn_config, args.net_seed


def get_data_config(in_config, old_stats=False):
    """Shortcut to control data configuration
        Note that the old experiment is HARDCODED!!!!!"""
    if old_stats:
        # get data stats from older runs to save runtime
        # TODO Update after getting run with zeros in mean edge coordinates!!
        old_experiment = WandbRunWrappper(system_info['wandb_username'],
            project_name='Garments-Reconstruction', 
            run_name='Pattern3D-capacity', run_id='3rzpdacg'
        )
        # NOTE data stats are ONLY correct for a specific data split, so these two need to go together
        split, _, data_config = old_experiment.data_info()
        data_config = {
            'standardize': data_config['standardize']  # the rest of the info is not needed here
        }
    else: # default split for reproducibility
        split = {'valid_percent': 10, 'test_percent': 10, 'random_seed': 10} 
        data_config = {}

    print(data_config)
    # update with freshly configured values
    data_config.update(in_config)

    print(data_config)

    return split, data_config


if __name__ == "__main__":
    
    # dataset_folder = 'data_1000_skirt_4_panels_200616-14-14-40'
    dataset_folder = 'data_1000_tee_200527-14-50-42_regen_200612-16-56-43'
    in_data_config, in_nn_config, net_seed = get_values_from_args()

    system_info = customconfig.Properties('./system.json')
    experiment = WandbRunWrappper(
        system_info['wandb_username'], 
        project_name='Garments-Reconstruction', 
        run_name='Pattern3D-edge-init', 
        run_id=None, no_sync=False)   # set run id to resume unfinished run!

    # NOTE this dataset involves point sampling SO data stats from previous runs might not be correct, especially if we change the number of samples
    split, data_config = get_data_config(in_data_config, old_stats=True)
    dataset = data.Garment3DPatternDataset(Path(system_info['datasets_path']) / dataset_folder, 
                                            data_config, gt_caching=True, feature_caching=True)

    trainer = Trainer(experiment, dataset, split, with_norm=True, with_visualization=True)  # only turn on visuals on custom garment data
    
    trainer.init_randomizer(net_seed)
    model = nets.GarmentPattern3D(
        dataset.config['element_size'], dataset.config['panel_len'], dataset.config['ground_truth_size'], dataset.config['standardize'], 
        in_nn_config
    )
    if hasattr(model, 'config'):
        trainer.update_config(NN=model.config)  # save NN configuration

    trainer.fit(model)  # Magic happens here

    # --------------- Final evaluation --------------
    dataset_wrapper = trainer.datawraper
    # save predictions
    prediction_path = dataset_wrapper.predict(model, save_to=Path(system_info['output']), sections=['validation', 'test'])
    print('Predictions saved to {}'.format(prediction_path))

    final_metrics = metrics.eval_metrics(model, dataset_wrapper, 'test', loop_loss=True)
    print ('Test metrics: {}'.format(final_metrics))
    experiment.add_statistic('test', final_metrics)

    # reflect predictions info in expetiment
    experiment.add_statistic('predictions_folder', prediction_path.name)
    experiment.add_artifact(prediction_path, dataset_wrapper.dataset.name, 'result')