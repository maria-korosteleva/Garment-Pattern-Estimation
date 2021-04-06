from pathlib import Path
import argparse
import numpy as np

# My modules
import customconfig
import data
import nets
import metrics
from trainer import Trainer
from experiment import WandbRunWrappper
import nn.evaluation_scripts.latent_space_vis as tsne_plot


def get_values_from_args():
    """command line arguments to control the run for running wandb Sweeps!"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()
    
    # Default values from run 3cyu4gef, best accuracy\speed after sweep y1mmngej

    # basic
    parser.add_argument('--mesh_samples_multiplier', '-m', help='number of samples per mesh as multiplier of 500', type=int, default=5)
    parser.add_argument('--net_seed', '-ns', help='random seed for net initialization', type=float, default=916143406)
    parser.add_argument('--obj_nametag', '-obj', help='substring to identify 3D model files to load', type=str, default='sim')
    # Pattern decoder
    parser.add_argument('--pattern_encoding_multiplier', '-pte', help='size of pattern encoding as multiplier of 10', type=int, default=9)
    parser.add_argument('--pattern_n_layers', '-ptl', help='number of layers in pattern decoder', type=int, default=2)
    parser.add_argument('--panel_encoding_multiplier', '-pe', help='size of panel encoding as multiplier of 10', type=int, default=10)
    parser.add_argument('--panel_n_layers', '-pl', help='number of layers in panel decoder', type=int, default=3)
    parser.add_argument('--pattern_decoder', '-rdec', help='type of pattern decoder module', type=str, default='LSTMDecoderModule')
    parser.add_argument('--panel_decoder', '-ldec', help='type of panel decoder module', type=str, default='LSTMDecoderModule')
    # stitches
    parser.add_argument('--st_tag_len', '-stlen', help='size of the stitch tag', type=int, default=3)
    parser.add_argument('--st_tag_margin', '-stmar', help='margin for stitch tags separation', type=float, default=0.3)
    parser.add_argument('--st_tag_hardnet', '-sthard', help='weather to use hardnet in stitch loss', type=int, default=0)

    # EdgeConv
    parser.add_argument('--conv_depth', '-cd', help='number of convolutional layers in EdgeConv', type=int, default=2)
    parser.add_argument('--k_multiplier', '-k', help='number of nearest neigbors for graph construction in EdgeConv as multiplier of 5', type=int, default=1)
    parser.add_argument('--ec_hidden_multiplier', '-ech', help='size of EdgeConv hidden layers as multiplier of 8', type=int, default=25)
    parser.add_argument('--ec_hidden_depth', '-echd', help='number of hidden layers in EdgeConv', type=int, default=2)
    parser.add_argument('--ec_feature_multiplier', '-ecf', help='size of EdgeConv feature on each conv as multiplier of 8', type=int, default=14)
    parser.add_argument('--ec_conv_aggr', '-ecca', help='type of feature aggregation in EdgeConv on edge level', type=str, default='max')
    parser.add_argument('--ec_global_aggr', '-ecga', help='type of feature aggregation in EdgeConv on graph level', type=str, default='mean')
    parser.add_argument('--ec_skip', '-ecsk', help='Wether to use skip connections in EdgeConv', type=int, default=0)
    parser.add_argument('--ec_gpool', '-ecgp', help='Wether to use graph pooling after convolution in EdgeConv', type=int, default=0)
    parser.add_argument('--ec_gpool_ratio', '-ecr', help='ratio of graph pooling in EdgeConv', type=float, default=0.1)

    args = parser.parse_args()
    print(args)

    data_config = {
        'mesh_samples': args.mesh_samples_multiplier * 500,
        'obj_filetag': args.obj_nametag
    }

    nn_config = {
        # pattern decoders
        'panel_encoding_size': args.panel_encoding_multiplier * 10,
        'panel_n_layers': args.panel_n_layers,
        'pattern_encoding_size': args.pattern_encoding_multiplier * 10,
        'pattern_n_layers': args.pattern_n_layers,
        'panel_decoder': args.panel_decoder,
        'pattern_decoder': args.pattern_decoder,

        # stitches
        'stitch_tag_dim': args.st_tag_len, 
        'stitch_tags_margin': args.st_tag_margin,
        'stitch_hardnet_version': args.st_tag_hardnet,

        # EdgeConv params
        'conv_depth': args.conv_depth, 
        'k_neighbors': args.k_multiplier * 5, 
        'EConv_hidden': args.ec_hidden_multiplier * 8, 
        'EConv_hidden_depth': args.ec_hidden_depth, 
        'EConv_feature': args.ec_feature_multiplier * 8, 
        'EConv_aggr': args.ec_conv_aggr, 
        'global_pool': args.ec_global_aggr, 
        'skip_connections': bool(args.ec_skip),
        'graph_pooling': bool(args.ec_gpool),
        'pool_ratio': args.ec_gpool_ratio  # only used when the graph pooling is enabled
    }

    return data_config, nn_config, args.net_seed


def get_data_config(in_config, old_stats=False):
    """Shortcut to control data configuration
        Note that the old experiment is HARDCODED!!!!!"""
    if old_stats:
        # get data stats from older runs to save runtime
        old_experiment = WandbRunWrappper(
            system_info['wandb_username'],
            project_name='Garments-Reconstruction', 
            run_name='multi-all-origin-fix', run_id='w9dgekok'
            # run_name='multi-all-split-data-stats', run_id='2m2w6uns'
        )
        # NOTE data stats are ONLY correct for a specific data split, so these two need to go together
        split, _, data_config = old_experiment.data_info()
        data_config = {
            # 'standardize': data_config['standardize'],
            'max_pattern_len': data_config['max_pattern_len'],
            'max_panel_len': data_config['max_panel_len'],
            'max_num_stitches': data_config['max_num_stitches'],  # the rest of the info is not needed here
            'max_datapoints_per_type': data_config['max_datapoints_per_type'] if 'max_datapoints_per_type' in data_config else None # keep the numbers too
        }
        # failsafety
        try:
            old_experiment.load_file('data_split.json', './wandb')
            split['filename'] = './wandb/data_split.json'
        except ValueError as e:  # if file not found, training will just proceed with generated split
            print(e)
            print('Experiment::Warning::Skipping..')

    else:  # default split for reproducibility
        # NOTE addining 'filename' property to the split will force the data to be loaded from that list, instead of being randomly generated
        split = {'valid_percent': 20, 'test_percent': 20, 'random_seed': 10, 'filename': './wandb/data_split.json'} 
        data_config = {'max_datapoints_per_type': 30}  # no upper limit of how much data to grab from each type

    # update with freshly configured values
    data_config.update(in_config)
    
    print(split)

    return split, data_config


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)  # for readability

    # dataset_folder = 'data_1000_skirt_4_panels_200616-14-14-40'
    dataset_folder = 'data_1000_tee_200527-14-50-42_regen_200612-16-56-43'
    dataset_list = [
        'data_uni_1000_tee_200527-14-50-42_regen_200612-16-56-43',
        'data_uni_1000_skirt_4_panels_200616-14-14-40', 
        'data_uni_1000_pants_straight_sides_210105-10-49-02'
    ]
    in_data_config, in_nn_config, net_seed = get_values_from_args()

    system_info = customconfig.Properties('./system.json')
    experiment = WandbRunWrappper(
        system_info['wandb_username'], 
        project_name='Test-Garments-Reconstruction', 
        run_name='datafolder-names', 
        run_id=None, no_sync=False)   # set run id to resume unfinished run!

    # NOTE this dataset involves point sampling SO data stats from previous runs might not be correct, especially if we change the number of samples
    split, data_config = get_data_config(in_data_config, old_stats=False)

    data_config.update(data_folders=dataset_list)
    # dataset = data.Garment2DPatternDataset(Path(system_info['datasets_path']) / dataset_folder, data_config, gt_caching=True, feature_caching=True)
    dataset = data.Garment3DPatternFullDataset(system_info['datasets_path'], 
                                               data_config, gt_caching=True, feature_caching=True)

    trainer = Trainer(experiment, dataset, split, with_norm=True, with_visualization=True)  # only turn on visuals on custom garment data

    trainer.init_randomizer(net_seed)
    # model = nets.GarmentPatternAE(dataset.config['element_size'], dataset.config['panel_len'], dataset.config['standardize'], 
    #     in_nn_config)
    model = nets.GarmentFullPattern3D(dataset.config, in_nn_config)
    model.with_quality_eval = True  # False to save compute time
    if hasattr(model, 'config'):
        trainer.update_config(NN=model.config)  # save NN configuration

    trainer.fit(model)  # Magic happens here

    # --------------- Final evaluation -- same as in test.py --------------
    # On the best-performing model
    try:
        model.load_state_dict(experiment.load_best_model()['model_state_dict'])
    except BaseException as e:  # not the best to catch all the exceptions here, but should work for most of cases foe now
        print(e)
        print('Train::Warning::Proceeding to evaluation with the current (final) model state')

    datawrapper = trainer.datawraper

    final_metrics = metrics.eval_metrics(model, datawrapper, 'validation')
    print('Validation metrics: {}'.format(final_metrics))
    experiment.add_statistic('valid_full', final_metrics)

    final_metrics = metrics.eval_metrics(model, datawrapper, 'valid_per_data_folder')
    print('Validation metrics breakdown: {}'.format(final_metrics))
    experiment.add_statistic('valid', final_metrics)

    final_metrics = metrics.eval_metrics(model, datawrapper, 'test')
    print('Test metrics: {}'.format(final_metrics))
    experiment.add_statistic('test_full', final_metrics)

    final_metrics = metrics.eval_metrics(model, datawrapper, 'test_per_data_folder')
    print('Test metrics breakdown: {}'.format(final_metrics))
    experiment.add_statistic('test', final_metrics)

    # save TSNE plot
    garment_enc, garment_classes, panel_enc, panel_classes = tsne_plot.get_encodings(model, datawrapper.get_loader('test'), dataset)

    tsne_plot.tsne_plot(garment_enc, garment_classes, experiment.local_path(), 'garments', dpi=150)
    tsne_plot.tsne_plot(panel_enc, panel_classes, experiment.local_path(), 'panels', dpi=150)

    # save predictions
    prediction_path = datawrapper.predict(model, save_to=Path(system_info['output']), sections=['validation', 'test'])
    print('Predictions saved to {}'.format(prediction_path))
    # reflect predictions info in expetiment
    experiment.add_statistic('predictions_folder', prediction_path.name)
    art_name = 'multi-data' if len(datawrapper.dataset.data_folders) > 1 else datawrapper.dataset.data_folders[0]

    experiment.add_artifact(prediction_path, art_name, 'result')

