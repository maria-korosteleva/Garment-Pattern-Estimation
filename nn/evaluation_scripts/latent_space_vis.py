"""
    This script uses results produced by save_latent_encodings.py
    Separation in two skipts was simply needed to experiment with visual style without re-running the net every time
"""

from pathlib import Path
import torch
import pickle
import numpy as np
from datetime import datetime

# visualization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib


# Do avoid a need for changing Evironmental Variables outside of this script
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# my
import customconfig
import data
import nets
from trainer import Trainer
from experiment import WandbRunWrappper


def load_model_loader(experiment, datasets_path, subset='test'):
    """
        Return NN model object and a test set loader accosiated with given experiment
    """
    # -------- data -------
    # data_config also contains the names of datasets to use
    split, batch_size, data_config = experiment.data_info()  # note that run is not initialized -- we use info from finished run
    data_config.update({'obj_filetag': 'sim'})  # , 'max_datapoints_per_type': 300})

    dataset = data.Garment3DPatternFullDataset(
        datasets_path, data_config, gt_caching=True, feature_caching=True)
    datawrapper = data.DatasetWrapper(dataset, known_split=split, batch_size=batch_size)
    test_loader = datawrapper.get_loader(subset)

    # ----- Model -------
    model_class = getattr(nets, experiment.NN_config()['model'])
    model = model_class(dataset.config, experiment.NN_config(), experiment.NN_config()['loss'])
    model.load_state_dict(experiment.load_best_model(device='cuda:0')['model_state_dict'])

    return test_loader, model, dataset


def get_encodings(model, loader, dataset, save_to=None):
    """
        Get latent space encodings on the test set from the given experiment (defines both dataset and the model)
        Save then to given folder if provided
    """

    # get all encodings
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    all_garment_encodings = []
    all_panel_encodings = []
    classes_garments = []
    classes_panels = []
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            garment_encodings = model.module.forward_encode(features)

            # NOTE some of the panel encodings actually correspond to empty panels 
            if hasattr(model.module, 'forward_panel_enc_from_3d'):
                panel_encodings, _ = model.module.forward_panel_enc_from_3d(features)
                panel_encodings = panel_encodings.view(-1, panel_encodings.shape[-1])  # flatten pattern dim
            else:
                panel_encodings = model.module.forward_pattern_decode(garment_encodings)

            all_garment_encodings.append(garment_encodings)
            all_panel_encodings.append(panel_encodings)

            classes_garments += batch['data_folder']
            for pattern_id in range(len(batch['data_folder'])):
                # copy the folder of pattern to every panel in it's patten
                classes_panels += [batch['data_folder'][pattern_id] for _ in range(model.module.max_pattern_size)]

    all_garment_encodings = torch.cat(all_garment_encodings).cpu().numpy()
    all_panel_encodings = torch.cat(all_panel_encodings).cpu().numpy()

    classes_garments = [dataset.data_folders_nicknames[data_folder] for data_folder in classes_garments]
    classes_panels = [dataset.data_folders_nicknames[data_folder] for data_folder in classes_panels]

    if save_to is not None:
        np.save(save_to / 'enc_garments.npy', all_garment_encodings)
        np.save(save_to / 'enc_panels.npy', all_panel_encodings)
        with open(save_to / 'data_folders_garments.pkl', 'wb') as fp:
            pickle.dump(classes_garments, fp)
        with open(save_to / 'data_folders_panels.pkl', 'wb') as fp:
            pickle.dump(classes_panels, fp)
    
    return all_garment_encodings, classes_garments, all_panel_encodings, classes_panels


def load_enc_from_files(encodings_folder, enc_tag='garments'):
    """
        Load serialized encodings for particular type of encoding
        Supported tags (aka types): {'garments', 'panels'}
    """

    all_encodings = np.load(encodings_folder / ('enc_' + enc_tag + '.npy'))
    with open(encodings_folder / ('data_folders_' + enc_tag + '.pkl'), 'rb') as fp:
        classes = pickle.load(fp)

    print(all_encodings.shape, len(classes))

    return all_encodings, classes


def tsne_plot(all_encodings, classes, num_components=2, save_to='./', name_tag='enc', interactive_mode=False, dpi=300):
    """
        Plot encodings using TSNE dimentionality reduction
    """
    if num_components != 2 and num_components != 3:
        raise NotImplementedError('Unsupported number ({}) of tsne components requested. Only 2D and 3D are supported'.format(
            num_components
        ))
    # https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    # https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html
    tsne = TSNE(n_components=num_components, random_state=0)
    enc_reduced = tsne.fit_transform(all_encodings)

    # ----- Visualize ------
    # update class labeling
    # This is hardcoded stuff because it's mostly needed for final presentation
    # TODO avoid hardcoding dataset names (or template names) -- picking up colors from dataset properties?
    mapping = {
        # train
        'tee_sleeveless': 'Sleeveless Shirts and dresses',
        'tee': 'Shirts and dresses',
        'jacket': 'Jacket',
        'jacket_hood': 'Open Hoody',
        'pants_straight_sides': 'Pants',
        'wb_pants_straight': 'Waistband pants',
        'jumpsuit_sleeveless': 'Sleeveless Jumpsuit',
        'skirt_4_panels': '4-panel Skirts',
        'skirt_2_panels': '2-panel Skirts',
        'skirt_8_panels': '8-panel Skirts',
        'dress_sleeveless': 'Sleeveless Dresses',
        'wb_dress_sleeveless': 'Sleeveless Waistband dresses'

        # test
    }
    classes = np.array([mapping.get(label, 'Unknown') for label in classes])

    # define colors
    colors = {
        # train
        'Shirts and dresses': (0.747, 0.236, 0.048),  # (190, 60, 12)
        'Sleevelss Shirts and dresses': (0.747, 0.236, 0.048),  # (190, 60, 12)
        'Jacket': (0.4793117642402649, 0.10461537539958954, 0.0),
        'Open Hoody': (0.746999979019165, 0.28518322110176086, 0.11503798514604568), 
        '8-panel Skirts': (0.048, 0.0290, 0.747),  # (12, 74, 190)
        '4-panel Skirts': (0.048, 0.0290, 0.747),  # (12, 74, 190)
        '2-panel Skirts': (0.24227915704250336, 0.6172128915786743, 1.0),  
        'Pants': (0.025, 0.354, 0.152),  # (6, 90. 39)
        'Sleeveless Jumpsuit': (0.6104, 0.3023, 0.0872),  # (105,52,15)  TODO DOuble check colors
        'Sleeveless Waistband dresses': (0.2, 0.007, 0.192),  
        'Sleeveless Dresses': (0.527, 0.0, 0.0),  # (134,0,0)
        'Waistband pants': (0.250900000333786, 0.3528999984264374, 0.13330000638961792), 

        # test
        'Dresses': (0.527, 0.0, 0.0),  # (134,0,0)
        'Jumpsuit': (0.6104, 0.3023, 0.0872),  # (105,52,15)

        # other
        'Waistband dresses': (0.2, 0.007, 0.192), 
        'Unknown': (0.15, 0.15, 0.15)
    }

    # global plot edges color setup
    # https://stackoverflow.com/questions/7778954/elegantly-changing-the-color-of-a-plot-frame-in-matplotlib/7944576
    base_color = '#737373'
    matplotlib.rc('axes', edgecolor=base_color)

    # plot
    if num_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:  # 2D
        fig, ax = plt.subplots()

    # plot data
    for label, color in colors.items():
        if len(enc_reduced[classes == label, 0] > 0):  # skip unused classes
            if num_components == 3:
                ax.scatter(
                    enc_reduced[classes == label, 0], enc_reduced[classes == label, 1], enc_reduced[classes == label, 2],
                    color=color, label=label,
                    edgecolors=None, alpha=0.5, s=17)
            else:
                ax.scatter(
                    enc_reduced[classes == label, 0], enc_reduced[classes == label, 1],
                    color=color, label=label,
                    edgecolors=None, alpha=0.5, s=17)
            if label == 'Unknown':
                print('TSNE_plot::Warning::Some labels are unknown and thus colored with Dark Grey')
    plt.legend()

    # Axes colors
    ax.tick_params(axis='x', colors=base_color)
    ax.tick_params(axis='y', colors=base_color)
    if num_components == 3:
        ax.tick_params(axis='z', colors=base_color)

    # plt.savefig('D:/MyDocs/GigaKorea/SIGGRAPH2021 submission materials/Latent space/tsne.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_to / ('tsne_' + name_tag + '_' + str(num_components) + 'd.pdf'), dpi=dpi, bbox_inches='tight')
    plt.savefig(save_to / ('tsne_' + name_tag + '_' + str(num_components) + 'd.jpg'), dpi=dpi, bbox_inches='tight')
    if interactive_mode:
        plt.show()
    
    print('Info::Saved TSNE plot for {}'.format(name_tag))


if __name__ == '__main__':
    system_info = customconfig.Properties('./system.json')
    from_experiment = True
    if from_experiment:
        experiment = WandbRunWrappper(
            system_info['wandb_username'],
            project_name='Test-Garments-Reconstruction', 
            run_name='attention-3d-ordered', 
            run_id='2sjhdio6')  # finished experiment

        if not experiment.is_finished():
            print('Warning::Evaluating unfinished experiment')

        out_folder = Path(system_info['output']) / ('tsne_' + experiment.run_name + '_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
        out_folder.mkdir(parents=True, exist_ok=True)

        # extract encodings
        data_loader, model, dataset = load_model_loader(experiment, system_info['datasets_path'], 'test')

        garment_enc, garment_classes, panel_enc, panel_classes = get_encodings(model, data_loader, dataset, out_folder)

    else:
        # if loading from file
        encodings_folder_name = 'tsne_tee-1000-800-server_210408-11-51-05'
        out_folder = Path(system_info['output']) / encodings_folder_name
        garment_enc, garment_classes = load_enc_from_files(out_folder, 'garments')
        panel_enc, panel_classes = load_enc_from_files(out_folder, 'panels')

    # save plots
    tsne_plot(garment_enc, garment_classes, 2, out_folder, 'garments', True, dpi=600)
    tsne_plot(garment_enc, garment_classes, 3, out_folder, 'garments', True, dpi=600)
    tsne_plot(panel_enc, panel_classes, 2, out_folder, 'panels', True, dpi=600)
    tsne_plot(panel_enc, panel_classes, 3, out_folder, 'panels', True, dpi=600)
