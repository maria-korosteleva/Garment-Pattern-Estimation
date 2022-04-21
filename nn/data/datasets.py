import json
import numpy as np
import os
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import igl
# import meshplot  # when uncommented, could lead to problems with wandb run syncing

# My modules
from customconfig import Properties
from pattern_converter import NNSewingPattern, InvalidPatternDefError
import data.transforms as transforms
from data.panel_classes import PanelClasses

# --------------------- Datasets -------------------------

class BaseDataset(Dataset):
    """
        * Implements base interface for my datasets
        * Implements routines for datapoint retrieval, structure & cashing 
        (agnostic of the desired feature & GT structure representation)
    """
    def __init__(self, root_dir, start_config={'data_folders': []}, gt_caching=False, feature_caching=False, in_transforms=[]):
        """Kind of Universal init for my datasets
            * Expects that all incoming datasets are located in the same root directory
            * The names of dataset_folders to use should be provided in start_config
                (defining it in dict allows to load data list as property from previous experiments)
            * if cashing is enabled, datapoints will stay stored in memory on first call to them: might speed up data processing by reducing file reads"""
        self.root_path = Path(root_dir)
        self.config = {}
        self.update_config(start_config)
        self.config['class'] = self.__class__.__name__

        self.data_folders = start_config['data_folders']
        self.data_folders_nicknames = dict(zip(self.data_folders, self.data_folders))
        
        # list of items = subfolders
        self.datapoints_names = []
        self.dataset_start_ids = []  # (folder, start_id) tuples -- ordered by start id
        for data_folder in self.data_folders:
            _, dirs, _ = next(os.walk(self.root_path / data_folder))
            # dataset name as part of datapoint name
            datapoints_names = [data_folder + '/' + name for name in dirs]
            self.dataset_start_ids.append((data_folder, len(self.datapoints_names)))
            clean_list = self._clean_datapoint_list(datapoints_names, data_folder)
            if ('max_datapoints_per_type' in self.config
                    and self.config['max_datapoints_per_type'] is not None
                    and len(clean_list) > self.config['max_datapoints_per_type']):
                # There is no need to do random sampling of requested number of datapoints
                # The sample sewing patterns are randomly generated in the first place without particulat order
                # hence, simple slicing of elements would be equivalent to sampling them randomly from the list
                clean_list = clean_list[:self.config['max_datapoints_per_type']] 

            self.datapoints_names += clean_list
        self.dataset_start_ids.append((None, len(self.datapoints_names)))  # add the total len as item for easy slicing
        self.config['size'] = len(self)

        # cashing setup
        self.gt_cached = {}
        self.gt_caching = gt_caching
        if gt_caching:
            print('BaseDataset::Info::Storing datapoints ground_truth info in memory')
        self.feature_cached = {}
        self.feature_caching = feature_caching
        if feature_caching:
            print('BaseDataset::Info::Storing datapoints feature info in memory')

        # Use default tensor transform + the ones from input
        self.transforms = [transforms.SampleToTensor()] + in_transforms

        # statistics already there --> need to apply it
        if 'standardize' in self.config:
            self.standardize()

        # FORDEBUG -- access all the datapoints to pre-populate the cache of the data
        # self._renew_cache()

        # in\out sizes
        self._estimate_data_shape()

    def save_to_wandb(self, experiment):
        """Save data cofiguration to current expetiment run"""
        # config
        experiment.add_config('dataset', self.config)


    def update_transform(self, transform):
        """apply new transform when loading the data"""
        raise NotImplementedError('BaseDataset:Error:current transform support is poor')
        # self.transform = transform

    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints_names)  

    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data. 
        Does not support list indexing"""
        
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        folder_elements = None  
        datapoint_name = self.datapoints_names[idx]

        features, ground_truth = self._get_sample_info(datapoint_name)
        
        folder, name = tuple(datapoint_name.split('/'))
        sample = {'features': features, 'ground_truth': ground_truth, 'name': name, 'data_folder': folder}

        # apply transfomations (equally to samples from files or from cache)
        for transform in self.transforms:
            sample = transform(sample)

        return sample

    def update_config(self, in_config):
        """Define dataset configuration:
            * to be part of experimental setup on wandb
            * Control obtainign values for datapoints"""
        self.config.update(in_config)

        # check the correctness of provided list of datasets
        if ('data_folders' not in self.config 
                or not isinstance(self.config['data_folders'], list)
                or len(self.config['data_folders']) == 0):
            raise RuntimeError('BaseDataset::Error::information on datasets (folders) to use is missing in the incoming config')

        self._update_on_config_change()

    def _drop_cache(self):
        """Clean caches of datapoints info"""
        self.gt_cached = {}
        self.feature_cached = {}

    def _renew_cache(self):
        """Flush the cache and re-fill it with updated information if any kind of caching is enabled"""
        self.gt_cached = {}
        self.feature_cached = {}
        if self.feature_caching or self.gt_caching:
            for i in range(len(self)):
                self[i]
            print('Data cached!')

    def indices_by_data_folder(self, index_list):
        """
            Separate provided indices according to dataset folders used in current dataset
        """
        ids_dict = dict.fromkeys(self.data_folders)  # consists of elemens of index_list
        ids_mapping_dict = dict.fromkeys(self.data_folders)  # reference to the elements in index_list
        index_list = np.array(index_list)
        
        # assign by comparing with data_folders start & end ids
        # enforce sort Just in case
        self.dataset_start_ids = sorted(self.dataset_start_ids, key=lambda idx: idx[1])

        for i in range(0, len(self.dataset_start_ids) - 1):
            ids_filter = (index_list >= self.dataset_start_ids[i][1]) & (index_list < self.dataset_start_ids[i + 1][1])
            ids_dict[self.dataset_start_ids[i][0]] = index_list[ids_filter]
            ids_mapping_dict[self.dataset_start_ids[i][0]] = np.flatnonzero(ids_filter)
        
        return ids_dict, ids_mapping_dict

    def subsets_per_datafolder(self, index_list=None):
        """
            Group given indices by datafolder and Return dictionary with Subset objects for each group.
            if None, a breakdown for the full dataset is given
        """
        if index_list is None:
            index_list = range(len(self))
        per_data, _ = self.indices_by_data_folder(index_list)
        breakdown = {}
        for folder, ids_list in per_data.items():
            breakdown[self.data_folders_nicknames[folder]] = Subset(self, ids_list)
        return breakdown

    def random_split_by_dataset(self, valid_per_type, test_per_type=0, split_type='count', with_breakdown=False):
        """
            Produce subset wrappers for training set, validations set, and test set (if requested)
            Supported split_types: 
                * split_type='percent' takes a given percentage of the data for evaluation subsets. It also ensures the equal 
                proportions of elements from each datafolder in each subset -- according to overall proportions of 
                datafolders in the whole dataset
                * split_type='count' takes this exact number of elements for the elevaluation subselts from each datafolder. 
                    Maximizes the use of training elements, and promotes fair evaluation on uneven datafolder distribution. 

        Note: 
            * it's recommended to shuffle the training set on batching as random permute is not 
              guaranteed in this function
        """

        if split_type != 'count' and split_type != 'percent':
            raise NotImplementedError('{}::Error::Unsupported split type <{}> requested'.format(
                self.__class__.__name__, split_type))

        train_ids, valid_ids, test_ids = [], [], []

        train_breakdown, valid_breakdown, test_breakdown = {}, {}, {}

        for dataset_id in range(len(self.data_folders)):
            folder_nickname = self.data_folders_nicknames[self.data_folders[dataset_id]]

            start_id = self.dataset_start_ids[dataset_id][1]
            end_id = self.dataset_start_ids[dataset_id + 1][1]   # marker of the dataset end included
            data_len = end_id - start_id

            permute = (torch.randperm(data_len) + start_id).tolist()

            # size defined according to requested type
            valid_size = int(data_len * valid_per_type / 100) if split_type == 'percent' else valid_per_type
            test_size = int(data_len * test_per_type / 100) if split_type == 'percent' else test_per_type

            train_size = data_len - valid_size - test_size

            train_sub, valid_sub = permute[:train_size], permute[train_size:train_size + valid_size]

            train_ids += train_sub
            valid_ids += valid_sub

            if test_size:
                test_sub = permute[train_size + valid_size:train_size + valid_size + test_size]
                test_ids += test_sub
            
            if with_breakdown:
                train_breakdown[folder_nickname] = Subset(self, train_sub)
                valid_breakdown[folder_nickname] = Subset(self, valid_sub)
                test_breakdown[folder_nickname] = Subset(self, test_sub) if test_size else None

        if with_breakdown:
            return (
                Subset(self, train_ids), 
                Subset(self, valid_ids),
                Subset(self, test_ids) if test_per_type else None, 
                train_breakdown, valid_breakdown, test_breakdown
            )
            
        return (
            Subset(self, train_ids), 
            Subset(self, valid_ids),
            Subset(self, test_ids) if test_size else None
        )

    def split_from_dict(self, split_dict, with_breakdown=False):
        """
            Reproduce the data split in the provided dictionary: 
            the elements of the currect dataset should play the same role as in provided dict
        """
        train_ids, valid_ids, test_ids = [], [], []
        train_breakdown, valid_breakdown, test_breakdown = {}, {}, {}

        training_datanames = set(split_dict['training'])
        valid_datanames = set(split_dict['validation'])
        test_datanames = set(split_dict['test'])
        
        for idx in range(len(self.datapoints_names)):
            if self.datapoints_names[idx] in training_datanames:  # usually the largest, so check first
                train_ids.append(idx)
            elif len(test_datanames) > 0 and self.datapoints_names[idx] in test_datanames:
                test_ids.append(idx)
            elif len(valid_datanames) > 0 and self.datapoints_names[idx] in valid_datanames:
                valid_ids.append(idx)
            # othervise, just skip

        if with_breakdown:
            train_breakdown = self.subsets_per_datafolder(train_ids)
            valid_breakdown = self.subsets_per_datafolder(valid_ids)
            test_breakdown = self.subsets_per_datafolder(test_ids)

            return (
                Subset(self, train_ids), 
                Subset(self, valid_ids),
                Subset(self, test_ids) if len(test_ids) > 0 else None,
                train_breakdown, valid_breakdown, test_breakdown
            )

        return (
            Subset(self, train_ids), 
            Subset(self, valid_ids),
            Subset(self, test_ids) if len(test_ids) > 0 else None
        )

    # -------- Data-specific functions --------
    def save_prediction_batch(self, *args, **kwargs):
        """Saves predicted params of the datapoint to the original data folder"""
        print('{}::Warning::No prediction saving is implemented'.format(self.__class__.__name__))

    def standardize(self, training=None):
        """Use element normalization/standardization based on stats from the training subset.
            Dataset is the object most aware of the datapoint structure hence it's the place to calculate & use the normalization.
            Uses either of two: 
            * training subset to calculate the data statistics -- the stats are only based on training subsection of the data
            * if stats info is already defined in config, it's used instead of calculating new statistics (usually when calling to restore dataset from existing experiment)
            configuration has a priority: if it's given, the statistics are NOT recalculated even if training set is provided:
                this allows to save some time
        """
        print('{}::Warning::No standardization is implemented'.format(self.__class__.__name__))

    def _clean_datapoint_list(self, datapoints_names, dataset_folder):
        """Remove non-datapoints subfolders, failing cases, etc. Children are to override this function when needed"""
        # See https://stackoverflow.com/questions/57042695/calling-super-init-gives-the-wrong-method-when-it-is-overridden
        return datapoints_names

    def _get_sample_info(self, datapoint_name):
        """
            Get features and Ground truth prediction for requested data example
        """
        if datapoint_name in self.gt_cached:  # might not be compatible with list indexing
            ground_truth = self.gt_cached[datapoint_name]
        else:
            ground_truth = np.array([0])

            if self.gt_caching:
                self.gt_cached[datapoint_name] = ground_truth
        
        if datapoint_name in self.feature_cached:
            features = self.feature_cached[datapoint_name]
        else:
            features = np.array([0])
            
            if self.feature_caching:  # save read values 
                self.feature_cached[datapoint_name] = features
        
        return features, ground_truth

    def _estimate_data_shape(self):
        """Get sizes/shapes of a datapoint for external references"""
        elem = self[0]
        feature_size = elem['features'].shape[0]
        gt_size = elem['ground_truth'].shape[0] if hasattr(elem['ground_truth'], 'shape') else None

        self.config['feature_size'], self.config['ground_truth_size'] = feature_size, gt_size

    def _update_on_config_change(self):
        """Update object inner state after config values have changed"""
        pass


class GarmentBaseDataset(BaseDataset):
    """Base class to work with data from custom garment datasets"""
        
    def __init__(self, root_dir, start_config={'data_folders': []}, gt_caching=False, feature_caching=False, in_transforms=[]):
        """
            Initialize dataset of garments with patterns
            * the list of dataset folders to use should be supplied in start_config!!!
            * the initial value is only given for reference
        """
        # initialize keys for correct dataset initialization
        if ('max_pattern_len' not in start_config 
                or 'max_panel_len' not in start_config
                or 'max_num_stitches' not in start_config):
            start_config.update(max_pattern_len=None, max_panel_len=None, max_num_stitches=None)
            pattern_size_initialized = False
        else:
            pattern_size_initialized = True

        if 'obj_filetag' not in start_config:
            start_config['obj_filetag'] = 'sim'  # look for objects with this tag in filename when loading 3D models

        if 'panel_classification' not in start_config:
            start_config['panel_classification'] = None
        self.panel_classifier = None

        super().__init__(root_dir, start_config, gt_caching=gt_caching, feature_caching=feature_caching, in_transforms=in_transforms)

        # To make sure the datafolder names are unique after updates
        all_nicks = self.data_folders_nicknames.values()
        if len(all_nicks) > len(set(all_nicks)):
            print('{}::Warning::Some data folder nicknames are not unique: {}. Reverting to the use of original folder names'.format(
                self.__class__.__name__, self.data_folders_nicknames
            ))
            self.data_folders_nicknames = dict(zip(self.data_folders, self.data_folders))

        # Load panel classifier
        if self.config['panel_classification'] is not None:
            self.panel_classifier = PanelClasses(self.config['panel_classification'])
            self.config.update(max_pattern_len=len(self.panel_classifier))

        # evaluate base max values for number of panels, number of edges in panels among pattern in all the datasets
        # NOTE: max_pattern_len is influcened by presence or abcense of self.panel_classifier
        if not pattern_size_initialized:
            num_panels = []
            num_edges_in_panel = []
            num_stitches = []
            for data_folder, start_id in self.dataset_start_ids:
                if data_folder is None: 
                    break

                datapoint = self.datapoints_names[start_id]
                folder_elements = [file.name for file in (self.root_path / datapoint).glob('*')]
                pattern_flat, _, _, stitches, _ = self._read_pattern(datapoint, folder_elements, with_stitches=True)  # just the edge info needed
                num_panels.append(pattern_flat.shape[0])
                num_edges_in_panel.append(pattern_flat.shape[1])  # after padding
                num_stitches.append(stitches.shape[1])

            self.config.update(
                max_pattern_len=max(num_panels),
                max_panel_len=max(num_edges_in_panel),
                max_num_stitches=max(num_stitches)
            )

        # to make sure that all the new datapoints adhere to evaluated structure!
        self._drop_cache() 
     
    def save_to_wandb(self, experiment):
        """Save data cofiguration to current expetiment run"""
        super().save_to_wandb(experiment)

        # dataset props files
        for dataset_folder in self.data_folders:
            try:
                shutil.copy(
                    self.root_path / dataset_folder / 'dataset_properties.json', 
                    experiment.local_wandb_path() / (dataset_folder + '_properties.json'))
            except FileNotFoundError:
                pass
        
        # panel classes
        if self.panel_classifier is not None:
            shutil.copy(
                self.panel_classifier.filename, 
                experiment.local_wandb_path() / ('panel_classes.json'))

        # param filter file
        if 'filter_by_params' in self.config and self.config['filter_by_params']:
            shutil.copy(
                self.config['filter_by_params'], 
                experiment.local_wandb_path() / ('param_filter.json'))
    
    # ------ Garment Data-specific basic functions --------
    def _clean_datapoint_list(self, datapoints_names, dataset_folder):
        """
            Remove all elements marked as failure from the provided list
            Updates the currect dataset nickname as a small sideeffect
        """

        try: 
            datapoints_names.remove(dataset_folder + '/renders')  # TODO read ignore list from props
        except ValueError:  # it's ok if there is no subfolder for renders
            pass

        try: 
            dataset_props = Properties(self.root_path / dataset_folder / 'dataset_properties.json')
        except FileNotFoundError:
            # missing dataset props file -- skip failure processing
            print(f'{self.__class__.__name__}::Warning::No `dataset_properties.json` found. Using all datapoints without filtering.')
            self.data_folders_nicknames[dataset_folder] = dataset_folder
            return datapoints_names

        if not dataset_props['to_subfolders']:
            raise NotImplementedError('Only working with datasets organized with subfolders')

        # NOTE A little side-effect here, since we are loading the dataset_properties anyway
        self.data_folders_nicknames[dataset_folder] = dataset_props['templates'].split('/')[-1].split('.')[0]

        fails_dict = dataset_props['sim']['stats']['fails']
        # TODO allow not to ignore some of the subsections
        for subsection in fails_dict:
            for fail in fails_dict[subsection]:
                try:
                    datapoints_names.remove(dataset_folder + '/' + fail)
                except ValueError:  # if fail was already removed based on previous failure subsection
                    pass
        
        # filter by parameters
        if 'filter_by_params' in self.config and self.config['filter_by_params']:
            datapoints_names = self.filter_by_params(
                self.config['filter_by_params'], dataset_folder, datapoints_names)

        return datapoints_names

    def filter_by_params(self, filter_file, dataset_folder, datapoint_names):
        """ Remove from considerstion datapoint that don't pass the parameter filter

            * filter_file -- path to .json file with allowed parameter ranges
            * dataset_folder -- data folder to filter
            * datapoint_names -- list of samples to apply filter to
        """
        with open(filter_file, 'r') as f:
            param_filters = json.load(f)

        final_list = []
        for datapoint_name in datapoint_names:
            pattern = NNSewingPattern(self.root_path / datapoint_name / 'specification.json')
            template_name = self.template_name(datapoint_name)
            to_add = True
            for param in param_filters[template_name]:
                value = pattern.parameters[param]['value']
                if value < param_filters[template_name][param][0] or value > param_filters[template_name][param][1]:
                    to_add = False
                    break
            if to_add:
                final_list.append(datapoint_name)
        
        print(f'{self.__class__.__name__}::Filtering::{dataset_folder}::{len(final_list)} of {len(datapoint_names)}')

        return final_list

    # ------------- Datapoints Utils --------------
    def template_name(self, datapoint_name):
        """Get name of the garment template from the path to the datapoint"""
        return self.data_folders_nicknames[datapoint_name.split('/')[0]]

    def _read_pattern(self, datapoint_name, folder_elements, 
                      pad_panels_to_len=None, pad_panel_num=None, pad_stitches_num=None,
                      with_placement=False, with_stitches=False, with_stitch_tags=False):
        """Read given pattern in tensor representation from file"""
        spec_list = [file for file in folder_elements if 'specification.json' in file]
        if not spec_list:
            raise RuntimeError('GarmentBaseDataset::Error::*specification.json not found for {}'.format(datapoint_name))
        
        pattern = NNSewingPattern(
            self.root_path / datapoint_name / spec_list[0], 
            panel_classifier=self.panel_classifier, 
            template_name=self.template_name(datapoint_name))
        return pattern.pattern_as_tensors(
            pad_panels_to_len, pad_panels_num=pad_panel_num, pad_stitches_num=pad_stitches_num,
            with_placement=with_placement, with_stitches=with_stitches, 
            with_stitch_tags=with_stitch_tags)

    def _pattern_from_tenzor(self, dataname, 
                             tenzor, 
                             rotations=None, translations=None, stitches=None,
                             std_config={}, supress_error=True):
        """Shortcut to create a pattern object from given tenzor and suppress exceptions if those arize"""
        if std_config and 'standardize' in std_config:
            tenzor = tenzor * self.config['standardize']['scale'] + self.config['standardize']['shift']

        pattern = NNSewingPattern(view_ids=False, panel_classifier=self.panel_classifier)
        pattern.name = dataname
        try: 
            pattern.pattern_from_tensors(
                tenzor, panel_rotations=rotations, panel_translations=translations, stitches=stitches,
                padded=True)   
        except (RuntimeError, InvalidPatternDefError) as e:
            if not supress_error:
                raise e
            print('Garment3DPatternDataset::Warning::{}: {}'.format(dataname, e))
            pass

        return pattern

    # -------- Generalized Utils -----
    def _unpad(self, element, tolerance=1.e-5):
        """Return copy of input element without padding from given element. Used to unpad edge sequences in pattern-oriented datasets"""
        # NOTE: might be some false removal of zero edges in the middle of the list.
        if torch.is_tensor(element):        
            bool_matrix = torch.isclose(element, torch.zeros_like(element), atol=tolerance)  # per-element comparison with zero
            selection = ~torch.all(bool_matrix, axis=1)  # only non-zero rows
        else:  # numpy
            selection = ~np.all(np.isclose(element, 0, atol=tolerance), axis=1)  # only non-zero rows
        return element[selection]

    def _get_distribution_stats(self, input_batch, padded=False):
        """Calculates mean & std values for the input tenzor along the last dimention"""

        input_batch = input_batch.view(-1, input_batch.shape[-1])
        if padded:
            input_batch = self._unpad(input_batch)  # remove rows with zeros

        # per dimention means
        mean = input_batch.mean(axis=0)
        # per dimention stds
        stds = ((input_batch - mean) ** 2).sum(0)
        stds = torch.sqrt(stds / input_batch.shape[0])

        return mean, stds

    def _get_norm_stats(self, input_batch, padded=False):
        """Calculate shift & scaling values needed to normalize input tenzor 
            along the last dimention to [0, 1] range"""
        input_batch = input_batch.view(-1, input_batch.shape[-1])
        if padded:
            input_batch = self._unpad(input_batch)  # remove rows with zeros

        # per dimention info
        min_vector, _ = torch.min(input_batch, dim=0)
        max_vector, _ = torch.max(input_batch, dim=0)
        scale = torch.empty_like(min_vector)

        # avoid division by zero
        for idx, (tmin, tmax) in enumerate(zip(min_vector, max_vector)): 
            if torch.isclose(tmin, tmax):
                scale[idx] = tmin if not torch.isclose(tmin, torch.zeros(1)) else 1.
            else:
                scale[idx] = tmax - tmin
        
        return min_vector, scale


class Garment3DPatternFullDataset(GarmentBaseDataset):
    """Dataset with full pattern definition as ground truth
        * it includes not only every panel outline geometry, but also 3D placement and stitches information
        Defines 3D samples from the point cloud as features
    """
    def __init__(self, root_dir, start_config={'data_folders': []}, gt_caching=False, feature_caching=False, in_transforms=[]):
        if 'mesh_samples' not in start_config:
            start_config['mesh_samples'] = 2000  # default value if not given -- a bettern gurantee than a default value in func params
        if 'point_noise_w' not in start_config:
            start_config['point_noise_w'] = 0  # default value if not given -- a bettern gurantee than a default value in func params
        
        # to cache segmentation mask if enabled
        self.segm_cached = {}

        super().__init__(root_dir, start_config, 
                         gt_caching=gt_caching, feature_caching=feature_caching, in_transforms=in_transforms)
        
        self.config.update(
            element_size=self[0]['ground_truth']['outlines'].shape[2],
            rotation_size=self[0]['ground_truth']['rotations'].shape[1],
            translation_size=self[0]['ground_truth']['translations'].shape[1],
            stitch_tag_size=self[0]['ground_truth']['stitch_tags'].shape[-1],
            explicit_stitch_tags=False
        )
    
    def standardize(self, training=None):
        """Use shifting and scaling for fitting data to interval comfortable for NN training.
            Accepts either of two inputs: 
            * training subset to calculate the data statistics -- the stats are only based on training subsection of the data
            * if stats info is already defined in config, it's used instead of calculating new statistics (usually when calling to restore dataset from existing experiment)
            configuration has a priority: if it's given, the statistics are NOT recalculated even if training set is provided
                => speed-up by providing stats or speeding up multiple calls to this function
        """
        print('Garment3DPatternFullDataset::Using data normalization for features & ground truth')

        if 'standardize' in self.config:
            print('{}::Using stats from config'.format(self.__class__.__name__))
            stats = self.config['standardize']
        elif training is not None:
            loader = DataLoader(training, batch_size=len(training), shuffle=False)
            for batch in loader:
                feature_shift, feature_scale = self._get_distribution_stats(batch['features'], padded=False)

                gt = batch['ground_truth']
                panel_shift, panel_scale = self._get_distribution_stats(gt['outlines'], padded=True)
                # NOTE mean values for panels are zero due to loop property 
                # panel components SHOULD NOT be shifted to keep the loop property intact 
                panel_shift[0] = panel_shift[1] = 0

                # Use min\scale (normalization) instead of Gaussian stats for translation
                # No padding as zero translation is a valid value
                transl_min, transl_scale = self._get_norm_stats(gt['translations'])
                rot_min, rot_scale = self._get_norm_stats(gt['rotations'])

                # stitch tags if given
                st_tags_min, st_tags_scale = self._get_norm_stats(gt['stitch_tags'])

                break  # only one batch out there anyway

            self.config['standardize'] = {
                'f_shift': feature_shift.cpu().numpy(), 
                'f_scale': feature_scale.cpu().numpy(),
                'gt_shift': {
                    'outlines': panel_shift.cpu().numpy(), 
                    'rotations': rot_min.cpu().numpy(),
                    'translations': transl_min.cpu().numpy(), 
                    'stitch_tags': st_tags_min.cpu().numpy()
                },
                'gt_scale': {
                    'outlines': panel_scale.cpu().numpy(), 
                    'rotations': rot_scale.cpu().numpy(),
                    'translations': transl_scale.cpu().numpy(),
                    'stitch_tags': st_tags_scale.cpu().numpy()
                }
            }
            stats = self.config['standardize']
        else:  # nothing is provided
            raise ValueError('Garment3DPatternFullDataset::Error::Standardization cannot be applied: supply either stats in config or training set to use standardization')

        # clean-up tranform list to avoid duplicates
        self.transforms = [t for t in self.transforms if not isinstance(t, transforms.GTtandartization) and not isinstance(t, transforms.FeatureStandartization)]

        self.transforms.append(transforms.GTtandartization(stats['gt_shift'], stats['gt_scale']))
        self.transforms.append(transforms.FeatureStandartization(stats['f_shift'], stats['f_scale']))

    # ----- Saving predictions -----
    def save_prediction_batch(
            self, predictions, datanames, data_folders, 
            save_to, features=None, weights=None, orig_folder_names=False, **kwargs):
        """ 
            Saving predictions on batched from the current dataset
            Saves predicted params of the datapoint to the requested data folder.
            Returns list of paths to files with prediction visualizations
            Assumes that the number of predictions matches the number of provided data names"""

        save_to = Path(save_to)
        prediction_imgs = []
        for idx, (name, folder) in enumerate(zip(datanames, data_folders)):

            # "unbatch" dictionary
            prediction = {}
            for key in predictions:
                prediction[key] = predictions[key][idx]

            # add values from GT if not present in prediction
            if (('order_matching' in self.config and self.config['order_matching'])
                    or 'origin_matching' in self.config and self.config['origin_matching']
                    or not self.gt_caching):
                print(f'{self.__class__.__name__}::Warning::Propagating '
                      'information from GT on prediction is not implemented in given context')
            else:
                gt = self.gt_cached[folder + '/' + name]
                for key in gt:
                    if key not in prediction:
                        prediction[key] = gt[key]

            # Transform to pattern object
            pattern = self._pred_to_pattern(prediction, name)

            # log gt number of panels
            if self.gt_caching:
                gt = self.gt_cached[folder + '/' + name]
                pattern.spec['properties']['correct_num_panels'] = gt['num_panels']

            # save prediction
            folder_nick = self.data_folders_nicknames[folder] if not orig_folder_names else folder

            try: 
                final_dir = pattern.serialize(save_to / folder_nick, to_subfolder=True, tag='_predicted_')
            except (RuntimeError, InvalidPatternDefError, TypeError) as e:
                print('Garment3DPatternDataset::Error::{} serializing skipped: {}'.format(name, e))
                continue
            
            final_file = pattern.name + '_predicted__pattern.png'
            prediction_imgs.append(Path(final_dir) / final_file)

            # copy originals for comparison
            for file in (self.root_path / folder / name).glob('*'):
                if ('.png' in file.suffix) or ('.json' in file.suffix):
                    shutil.copy2(str(file), str(final_dir))

            # save point samples if given 
            if features is not None:
                shift = self.config['standardize']['f_shift']
                scale = self.config['standardize']['f_scale']
                point_cloud = features[idx] * scale + shift

                np.savetxt(
                    save_to / folder_nick / name / (name + '_point_cloud.txt'), 
                    point_cloud
                )
            # save per-point weights if given
            if 'att_weights' in prediction:
                np.savetxt(
                    save_to / folder_nick / name / (name + '_att_weights.txt'), 
                    prediction['att_weights'].cpu().numpy()
                )
                    
        return prediction_imgs

    def _pred_to_pattern(self, prediction, dataname):
        """Convert given predicted value to pattern object
        """

        # undo standardization  (outside of generinc conversion function due to custom std structure)
        gt_shifts = self.config['standardize']['gt_shift']
        gt_scales = self.config['standardize']['gt_scale']
        for key in gt_shifts:
            if key == 'stitch_tags' and not self.config['explicit_stitch_tags']:  
                # ignore stitch tags update if explicit tags were not used
                continue
            prediction[key] = prediction[key].cpu().numpy() * gt_scales[key] + gt_shifts[key]

        if 'stitches' in prediction:  # if somehow prediction already has an answer
            stitches = prediction['stitches']
        else:  # stitch tags to stitch list
            stitches = self.tags_to_stitches(
                torch.from_numpy(prediction['stitch_tags']) if isinstance(prediction['stitch_tags'], np.ndarray) else prediction['stitch_tags'],
                prediction['free_edges_mask']
            )

        return self._pattern_from_tenzor(
            dataname, 
            prediction['outlines'], prediction['rotations'], prediction['translations'], 
            stitches, 
            std_config={}, supress_error=True)

    # ----- Sample -----
    def _get_sample_info(self, datapoint_name):
        """
            Get features and Ground truth prediction for requested data example
        """
        folder_elements = [file.name for file in (self.root_path / datapoint_name).glob('*')]  # all files in this directory

        # features -- points
        if datapoint_name in self.feature_cached:
            points = self.feature_cached[datapoint_name]
            segm = self.segm_cached[datapoint_name]
        else:
            points, verts = self._sample_points(datapoint_name, folder_elements)

            # Segmentation
            segm = self._point_classes_from_mesh(
                points, verts, datapoint_name, folder_elements)
            
            if self.feature_caching:  # save read values 
                self.feature_cached[datapoint_name] = points
                self.segm_cached[datapoint_name] = segm
        
        # GT -- pattern and segmentation
        if datapoint_name in self.gt_cached:  # might not be compatible with list indexing
            ground_truth = self.gt_cached[datapoint_name]
        else:
            ground_truth = self._get_pattern_ground_truth(datapoint_name, folder_elements)
            ground_truth['segmentation'] = segm

            if self.gt_caching:
                self.gt_cached[datapoint_name] = ground_truth

        return points, ground_truth
      
    def _get_pattern_ground_truth(self, datapoint_name, folder_elements):
        """Get the pattern representation with 3D placement"""
        pattern, num_edges, num_panels, rots, tranls, stitches, num_stitches, stitch_tags = self._read_pattern(
            datapoint_name, folder_elements, 
            pad_panels_to_len=self.config['max_panel_len'],
            pad_panel_num=self.config['max_pattern_len'],
            pad_stitches_num=self.config['max_num_stitches'],
            with_placement=True, with_stitches=True, with_stitch_tags=True)
        free_edges_mask = self.free_edges_mask(pattern, stitches, num_stitches)
        empty_panels_mask = self._empty_panels_mask(num_edges)  # useful for evaluation


        return {
            'outlines': pattern, 'num_edges': num_edges,
            'rotations': rots, 'translations': tranls, 
            'num_panels': num_panels, 'empty_panels_mask': empty_panels_mask, 'num_stitches': num_stitches,
            'stitches': stitches, 'free_edges_mask': free_edges_mask, 'stitch_tags': stitch_tags}

    # ----- Mesh tools -----
    def _sample_points(self, datapoint_name, folder_elements):
        """Make a sample from the 3d surface from a given datapoint files

            Returns: sampled points and vertices of original mesh
        
        """
        obj_list = [file for file in folder_elements if self.config['obj_filetag'] in file and '.obj' in file]
        if not obj_list:
            raise RuntimeError('Dataset:Error: geometry file *{}*.obj not found for {}'.format(self.config['obj_filetag'], datapoint_name))
        
        verts, faces = igl.read_triangle_mesh(str(self.root_path / datapoint_name / obj_list[0]))
        points = self.sample_mesh_points(self.config['mesh_samples'], verts, faces)

        # add gaussian noise
        if self.config['point_noise_w']:
            points += np.random.normal(loc=0.0, scale=self.config['point_noise_w'], size=points.shape)

        # Debug
        # if 'skirt_4_panels_00HUVRGNCG' in datapoint_name:
        #     meshplot.offline()
        #     meshplot.plot(points, c=points[:, 0], shading={"point_size": 3.0})
        return points, verts

    @staticmethod
    def sample_mesh_points(num_points, verts, faces):
        """A routine to sample requested number of points from a given mesh
            Returns points in world coordinates"""

        barycentric_samples, face_ids = igl.random_points_on_mesh(num_points, verts, faces)
        face_ids[face_ids >= len(faces)] = len(faces) - 1  # workaround for https://github.com/libigl/libigl/issues/1531

        # convert to world coordinates
        points = np.empty(barycentric_samples.shape)
        for i in range(len(face_ids)):
            face = faces[face_ids[i]]
            barycentric_coords = barycentric_samples[i]
            face_verts = verts[face]
            points[i] = np.dot(barycentric_coords, face_verts)

        return points

    def _point_classes_from_mesh(self, points, verts, datapoint_name, folder_elements):
        """Map segmentation from original mesh to sampled points"""

        # load segmentation
        seg_path_list = [file for file in folder_elements if self.config['obj_filetag'] in file and 'segmentation.txt' in file]

        with open(str(self.root_path / datapoint_name / seg_path_list[0]), 'r') as f:
            vert_labels = np.array([line.rstrip() for line in f])  # remove \n
        map_list, _, _ = igl.snap_points(points, verts)

        if len(verts) > len(vert_labels):
            point_segmentation = np.zeros(len(map_list))
            print(f'{self.__class__.__name__}::{datapoint_name}::WARNING::Not enough segmentation labels -- {len(vert_labels)} for {len(verts)} vertices. Setting segmenations to zero')

            return point_segmentation.astype(np.int64)

        point_segmentation_names = vert_labels[map_list]

        # find those that map to stitches and assign them the closest panel label
        # Also doing this for occasional 'None's 
        stitch_points_ids = np.logical_or(
            point_segmentation_names == 'stitch', point_segmentation_names == 'None')
        non_stitch_points_ids = np.logical_and(
            point_segmentation_names != 'stitch', point_segmentation_names != 'None')

        map_stitches, _, _ = igl.snap_points(points[stitch_points_ids], points[non_stitch_points_ids])

        non_stitch_points_ids = np.flatnonzero(non_stitch_points_ids)
        point_segmentation_names[stitch_points_ids] = point_segmentation_names[non_stitch_points_ids[map_stitches]]

        # Map class names to int ids of loaded classes!
        if self.panel_classifier is not None:
            point_segmentation = self.panel_classifier.map(
                self.template_name(datapoint_name), point_segmentation_names)
        else:
            # assign unique ids within given list
            unique_names = np.unique(point_segmentation_names)
            unique_dict = {name: idx for idx, name in enumerate(unique_names)}
            point_segmentation = np.empty(len(point_segmentation_names))
            for idx, name in enumerate(point_segmentation_names):
                point_segmentation[idx] = unique_dict[name]

        return point_segmentation.astype(np.int64)   # type conversion for PyTorch NLLoss

    def _empty_panels_mask(self, num_edges):
        """Empty panels as boolean mask"""

        mask = np.zeros(len(num_edges), dtype=np.bool)
        mask[num_edges == 0] = True

        return mask

    # ----- Stitches tools -----
    @staticmethod
    def tags_to_stitches(stitch_tags, free_edges_score):
        """
        Convert per-edge per panel stitch tags into the list of connected edge pairs
        NOTE: expects inputs to be torch tensors, numpy is not supported
        """
        flat_tags = stitch_tags.view(-1, stitch_tags.shape[-1])  # with pattern-level edge ids
        
        # to edge classes from logits
        flat_edges_score = free_edges_score.view(-1) 
        flat_edges_mask = torch.round(torch.sigmoid(flat_edges_score)).type(torch.BoolTensor)

        # filter free edges
        non_free_mask = ~flat_edges_mask
        non_free_edges = torch.nonzero(non_free_mask, as_tuple=False).squeeze(-1)  # mapping of non-free-edges ids to full edges list id
        if not any(non_free_mask) or non_free_edges.shape[0] < 2:  # -> no stitches
            print('Garment3DPatternFullDataset::Warning::no non-zero stitch tags detected')
            return torch.tensor([])

        # Check for even number of tags
        if len(non_free_edges) % 2:  # odd => at least one of tags is erroneously non-free
            # -> remove the edge that is closest to free edges class from comparison
            to_remove = flat_edges_score[non_free_mask].argmax()  # the higer the score, the closer the edge is to free edges
            non_free_mask[non_free_edges[to_remove]] = False
            non_free_edges = torch.nonzero(non_free_mask, as_tuple=False).squeeze(-1)

        # Now we have even number of tags to match
        num_non_free = len(non_free_edges) 
        dist_matrix = torch.cdist(flat_tags[non_free_mask], flat_tags[non_free_mask])

        # remove self-distance on diagonal & lower triangle elements (duplicates)
        tril_ids = torch.tril_indices(num_non_free, num_non_free)
        dist_matrix[tril_ids[0], tril_ids[1]] = float('inf')

        # pair egdes by min distance to each other starting by the closest pair
        stitches = []
        for _ in range(num_non_free // 2):  # this many pair to arrange
            to_match_idx = dist_matrix.argmin()  # current global min is also a best match for the pair it's calculated for!
            row = to_match_idx // dist_matrix.shape[0]
            col = to_match_idx % dist_matrix.shape[0]
            stitches.append([non_free_edges[row], non_free_edges[col]])

            # exlude distances with matched edges from further consideration
            dist_matrix[row, :] = float('inf')
            dist_matrix[:, row] = float('inf')
            dist_matrix[:, col] = float('inf')
            dist_matrix[col, :] = float('inf')
        
        if torch.isfinite(dist_matrix).any():
            raise ValueError('Garment3DPatternFullDataset::Error::Tags-to-stitches::Number of stitches {} & dist_matrix shape {} mismatch'.format(
                num_non_free / 2, dist_matrix.shape))

        return torch.tensor(stitches).transpose(0, 1).to(stitch_tags.device) if len(stitches) > 0 else torch.tensor([])

    @staticmethod
    def free_edges_mask(pattern, stitches, num_stitches):
        """
        Construct the mask to identify edges that are not connected to any other
        """
        mask = np.ones((pattern.shape[0], pattern.shape[1]), dtype=np.bool)
        max_edge = pattern.shape[1]

        for side in stitches[:, :num_stitches]:  # ignore the padded part
            for edge_id in side:
                mask[edge_id // max_edge][edge_id % max_edge] = False
        
        return mask


class GarmentStitchPairsDataset(GarmentBaseDataset):
    """
        Dataset targets the task of predicting if a particular pair of edges is connected by a stitch or not
    """
    def __init__(
            self, 
            root_dir, start_config={'data_folders': []}, 
            gt_caching=False, feature_caching=False, in_transforms=[], 
            filter_correct_n_panels=False):
        if gt_caching or feature_caching:
            gt_caching = feature_caching = True  # ensure that both are simulataneously True or False
        
        self.filter_correct_n_panels = filter_correct_n_panels
        # data-specific defaults
        init_config = {
            'data_folders': [],
            'random_pairs_mode': True,   # False to use all possible pairs
            'stitched_edge_pairs_num': 100,
            'non_stitched_edge_pairs_num': 100,
            'shuffle_pairs': False, 
            'shuffle_pairs_order': False
        }
        init_config.update(start_config)  # values from input

        super().__init__(root_dir, init_config, 
                         gt_caching=gt_caching, feature_caching=feature_caching, in_transforms=in_transforms)

        self.config.update(
            element_size=self[0]['features'].shape[-1],
        )
        
        self.correctness_mask = []

    def standardize(self, training=None):
        """Use shifting and scaling for fitting data to interval comfortable for NN training.
            Accepts either of two inputs: 
            * training subset to calculate the data statistics -- the stats are only based on training subsection of the data
            * if stats info is already defined in config, it's used instead of calculating new statistics (usually when calling to restore dataset from existing experiment)
            configuration has a priority: if it's given, the statistics are NOT recalculated even if training set is provided
                => speed-up by providing stats or speeding up multiple calls to this function
        """
        print('{}::Using data normalization for features & ground truth'.format(self.__class__.__name__))

        if 'standardize' in self.config:
            print('{}::Using stats from config'.format(self.__class__.__name__))
            stats = self.config['standardize']
        elif training is not None:
            loader = DataLoader(training, batch_size=len(training), shuffle=False)
            for batch in loader:
                feature_shift, feature_scale = self._get_norm_stats(batch['features'], padded=False)
                break  # only one batch out there anyway

            self.config['standardize'] = {
                'f_shift': feature_shift.cpu().numpy(), 
                'f_scale': feature_scale.cpu().numpy(),
            }
            stats = self.config['standardize']
        else:  # nothing is provided
            raise ValueError('Garment3DPatternFullDataset::Error::Standardization cannot be applied: supply either stats in config or training set to use standardization')

        # clean-up tranform list to avoid duplicates
        self.transforms = [t for t in self.transforms if not isinstance(t, transforms.GTtandartization) and not isinstance(t, transforms.FeatureStandartization)]

        self.transforms.append(transforms.FeatureStandartization(stats['f_shift'], stats['f_scale']))


    def save_prediction_batch(
            self, predictions, datanames, data_folders, save_to, 
            model=None, orig_folder_names=False, **kwargs):
        """ 
            Saving predictions on batch from the current dataset based on given model
            Saves predicted params of the datapoint to the requested data folder.
            Returns list of paths to files with prediction visualizations
        """

        save_to = Path(save_to)
        prediction_imgs = []
        for idx, (name, folder) in enumerate(zip(datanames, data_folders)):

            # Load corresponding pattern
            folder_elements = [file.name for file in (self.root_path / folder / name).glob('*')]  # all files in this directory
            spec_list = [file for file in folder_elements if 'specification.json' in file]
            if not spec_list:
                print('{}::Error::{} serializing skipped: *specification.json not found'.format(
                    self.__class__.__name__, name))
                continue
            
            pattern = NNSewingPattern(self.root_path / folder / name / spec_list[0])

            # find stitches
            pattern.stitches_from_pair_classifier(model, self.config['standardize'])

            # save prediction
            # TODO Move to separate fucntion (for all datasets)
            folder_nick = self.data_folders_nicknames[folder] if not orig_folder_names else folder

            try: 
                final_dir = pattern.serialize(save_to / folder_nick, to_subfolder=True, tag='_predicted_')
            except (RuntimeError, InvalidPatternDefError, TypeError) as e:
                print('{}::Error::{} serializing skipped: {}'.format(self.__class__.__name__, name, e))
                continue
            
            final_file = pattern.name + '_predicted__pattern.png'
            prediction_imgs.append(Path(final_dir) / final_file)

            # copy originals for comparison
            for file in (self.root_path / folder / name).glob('*'):
                if ('.png' in file.suffix) or ('.json' in file.suffix):
                    shutil.copy2(str(file), str(final_dir))
                    
        return prediction_imgs


    def _get_sample_info(self, datapoint_name):
        """
            Get features and Ground truth prediction for requested data example
        """
        if datapoint_name in self.gt_cached:  # autpmatically means that features are cashed too
            ground_truth = self.gt_cached[datapoint_name]
            features = self.feature_cached[datapoint_name]

            return features, ground_truth

        # Get stitch pairs & mask from spec
        folder_elements = [file.name for file in (self.root_path / datapoint_name).glob('*')]  # all files in this directory
        spec_list = [file for file in folder_elements if 'specification.json' in file]
        if not spec_list:
            raise RuntimeError('GarmentBaseDataset::Error::*specification.json not found for {}'.format(datapoint_name))
        
        # Load from prediction if exists
        predicted_list = [file for file in spec_list if 'predicte' in file]
        spec = predicted_list[0] if len(predicted_list) > 0 else spec_list[0]

        pattern = NNSewingPattern(self.root_path / datapoint_name / spec)

        if self.config['random_pairs_mode']:
            features, ground_truth = pattern.stitches_as_3D_pairs(
                self.config['stitched_edge_pairs_num'], self.config['non_stitched_edge_pairs_num'],
                self.config['shuffle_pairs'], self.config['shuffle_pairs_order'])
        else:
            features, _, ground_truth = pattern.all_edge_pairs()
        
        # save elements
        if self.gt_caching and self.feature_caching:
            self.gt_cached[datapoint_name] = ground_truth
            self.feature_cached[datapoint_name] = features
            
        
        return features, ground_truth

    def _clean_datapoint_list(self, datapoints_names, dataset_folder):
        super()._clean_datapoint_list(datapoints_names, dataset_folder)
        
        final_list = []
        for datapoint_name in datapoints_names:
            folder_elements = [file.name for file in (self.root_path / datapoint_name).glob('*')]  # all files in this directory
            spec_list = [file for file in folder_elements if 'specification.json' in file]
            if not spec_list:
                raise RuntimeError('GarmentBaseDataset::Error::*specification.json not found for {}'.format(datapoint_name))
            
            # Load from prediction if exists
            predicted_list = [file for file in spec_list if 'predicte' in file]
            spec = predicted_list[0] if len(predicted_list) > 0 else spec_list[0]
            pattern = NNSewingPattern(self.root_path / datapoint_name / spec)

            if not len(pattern.pattern['stitches']):
                print(f'{self.__class__.__name__}::ERROR::{datapoint_name}::has no stitches')
                continue
            if self.filter_correct_n_panels:
                correct_num_panels = pattern.spec['properties']['correct_num_panels']
                actual_num_panels = len(pattern.pattern['panels'])
                if correct_num_panels != actual_num_panels:
                    continue
            
            final_list.append(datapoint_name)
        return final_list


if __name__ == "__main__":
    from wrapper import DatasetWrapper

    # DEBUG Basic debug of the data classes
    system = Properties('./system.json')
    dataset = Garment3DPatternFullDataset(system['datasets_path'], {
        'data_folders': [
            'tee_2300',
            'skirt_4_panels_1000'
        ]
    })

    print(len(dataset), dataset.config)

    datawrapper = DatasetWrapper(dataset)
    datawrapper.new_split(10, 10, 3000)
    datawrapper.standardize_data()

    print(dataset.config['standardize'])
