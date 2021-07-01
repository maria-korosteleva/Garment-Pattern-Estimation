import json
import numpy as np
import os
from pathlib import Path
import shutil
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import igl
# import meshplot  # when uncommented, could lead to problems with wandb run syncing

# My modules
from customconfig import Properties
from pattern_converter import NNSewingPattern, InvalidPatternDefError


# ---------------------- Main Wrapper ------------------
class DatasetWrapper(object):
    """Resposible for keeping dataset, its splits, loaders & processing routines.
        Allows to reproduce earlier splits
    """
    def __init__(self, in_dataset, known_split=None, batch_size=None, shuffle_train=True):
        """Initialize wrapping around provided dataset. If splits/batch_size is known """

        self.dataset = in_dataset
        self.data_section_list = ['full', 'train', 'validation', 'test']

        self.training = in_dataset
        self.validation = None
        self.test = None
        self.full_per_datafolder = None

        self.batch_size = None
        self.loader_full = None
        self.loader_train = None
        self.loader_validation = None
        self.loader_test = None

        self.split_info = {
            'random_seed': None, 
            'valid_per_type': None, 
            'test_per_type': None
        }

        if known_split is not None:
            self.load_split(known_split)
        if batch_size is not None:
            self.batch_size = batch_size
            self.new_loaders(batch_size, shuffle_train)
    
    def get_loader(self, data_section='full'):
        """Return loader that corresponds to given data section. None if requested loader does not exist"""
        if data_section == 'full':
            return self.loader_full
        elif data_section == 'full_per_data_folder':
            return self.loader_full_per_data
        elif data_section == 'train':
            return self.loader_train
        elif data_section == 'test':
            return self.loader_test
        elif data_section == 'validation':
            return self.loader_validation
        elif data_section == 'valid_per_data_folder':
            return self.loader_validation_per_data
        elif data_section == 'test_per_data_folder':
            return self.loader_test_per_data
        
        raise ValueError('DataWrapper::requested loader on unknown data section {}'.format(data_section))

    def new_loaders(self, batch_size=None, shuffle_train=True):
        """Create loaders for current data split. Note that result depends on the random number generator!"""
        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is None:
            raise RuntimeError('DataWrapper:Error:cannot create loaders: batch_size is not set')

        self.loader_train = DataLoader(self.training, self.batch_size, shuffle=shuffle_train)
        # no need for breakdown per datafolder for training -- for now

        self.loader_validation = DataLoader(self.validation, self.batch_size) if self.validation else None
        self.loader_validation_per_data = self._loaders_dict(self.validation_per_datafolder, self.batch_size) if self.validation else None
        # loader with per-data folder examples for visualization
        if self.validation:
            # indices_breakdown = self.dataset.indices_by_data_folder(self.validation.indices)
            single_sample_ids = [folder_ids.indices[0] for folder_ids in self.validation_per_datafolder.values()]
            self.loader_valid_single_per_data = DataLoader(
                Subset(self.dataset, single_sample_ids), batch_size=self.batch_size, shuffle=False) 
        else:
            self.loader_valid_single_per_data = None

        self.loader_test = DataLoader(self.test, self.batch_size) if self.test else None
        self.loader_test_per_data = self._loaders_dict(self.test_per_datafolder, self.batch_size) if self.test else None

        self.loader_full = DataLoader(self.dataset, self.batch_size)
        if self.full_per_datafolder is None:
            self.full_per_datafolder = self.dataset.subsets_per_datafolder()
        self.loader_full_per_data = self._loaders_dict(self.full_per_datafolder, self.batch_size)

        return self.loader_train, self.loader_validation, self.loader_test

    def _loaders_dict(self, subsets_dict, batch_size, shuffle=False):
        """Create loaders for all subsets in dict"""
        loaders_dict = {}
        for name, subset in subsets_dict.items():
            loaders_dict[name] = DataLoader(subset, batch_size, shuffle=shuffle)
        return loaders_dict

    # -------- Reproducibility ---------------
    def new_split(self, valid, test=None, random_seed=None):
        """Creates train/validation or train/validation/test splits
            depending on provided parameters
            """
        self.split_info['random_seed'] = random_seed if random_seed else int(time.time())
        self.split_info.update(valid_per_type=valid, test_per_type=test, type='count')
        
        return self.load_split()

    def load_split(self, split_info=None, batch_size=None):
        """Get the split by provided parameters. Can be used to reproduce splits on the same dataset.
            NOTE this function re-initializes torch random number generator!
        """
        if split_info:
            self.split_info = split_info

        if 'random_seed' not in self.split_info or self.split_info['random_seed'] is None:
            self.split_info['random_seed'] = int(time.time())
        torch.manual_seed(self.split_info['random_seed'])

        # if file is provided
        if 'filename' in self.split_info and self.split_info['filename'] is not None:
            print('DataWrapper::Loading data split from {}'.format(self.split_info['filename']))
            with open(self.split_info['filename'], 'r') as f_json:
                split_dict = json.load(f_json)
            self.training, self.validation, self.test, self.training_per_datafolder, self.validation_per_datafolder, self.test_per_datafolder = self.dataset.split_from_dict(
                split_dict, 
                with_breakdown=True)
        else:
            keys_required = ['test_per_type', 'valid_per_type', 'type']
            if any([key not in self.split_info for key in keys_required]):
                raise ValueError('Specified split information is not full: {}. It needs to contain: {}'.format(split_info, keys_required))
            print('DataWrapper::Loading data split from split config: {}: valid per type {} / test per type {}'.format(
                self.split_info['type'], self.split_info['valid_per_type'], self.split_info['test_per_type']))
            self.training, self.validation, self.test, self.training_per_datafolder, self.validation_per_datafolder, self.test_per_datafolder = self.dataset.random_split_by_dataset(
                self.split_info['valid_per_type'], 
                self.split_info['test_per_type'],
                self.split_info['type'],
                with_breakdown=True)

        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is not None:
            self.new_loaders()  # s.t. loaders could be used right away

        print('DatasetWrapper::Dataset split: {} / {} / {}'.format(
            len(self.training) if self.training else None, 
            len(self.validation) if self.validation else None, 
            len(self.test) if self.test else None))
        self.split_info['size_train'] = len(self.training) if self.training else 0
        self.split_info['size_valid'] = len(self.validation) if self.validation else 0
        self.split_info['size_test'] = len(self.test) if self.test else 0
        
        self.print_subset_stats(self.training_per_datafolder, len(self.training), 'Training', log_to_config=True)
        self.print_subset_stats(self.validation_per_datafolder, len(self.validation), 'Validation')
        self.print_subset_stats(self.test_per_datafolder, len(self.test), 'Test')

        return self.training, self.validation, self.test

    def print_subset_stats(self, subset_breakdown_dict, total_len, subset_name='', log_to_config=False):
        """Print stats on the elements of each datafolder contained in given subset"""
        # gouped by data_folders
        if not total_len:
            print('{}::Warning::Subset {} is empty, no stats printed'.format(self.__class__.__name__, subset_name))
            return
        self.split_info[subset_name] = {}
        message = ''
        for data_folder, subset in subset_breakdown_dict.items():
            if log_to_config:
                self.split_info[subset_name][data_folder] = len(subset)
            message += '{} : {:.1f}%;\n'.format(data_folder, 100 * len(subset) / total_len)
        
        print('DatasetWrapper::{} subset breakdown::\n{}'.format(subset_name, message))

    def save_to_wandb(self, experiment):
        """Save current data info to the wandb experiment"""
        # Split
        experiment.add_config('data_split', self.split_info)
        # save serialized split s.t. it's loaded to wandb
        split_datanames = {}
        split_datanames['training'] = [self.dataset.datapoints_names[idx] for idx in self.training.indices]
        split_datanames['validation'] = [self.dataset.datapoints_names[idx] for idx in self.validation.indices]
        split_datanames['test'] = [self.dataset.datapoints_names[idx] for idx in self.test.indices]
        with open(experiment.local_path() / 'data_split.json', 'w') as f_json:
            json.dump(split_datanames, f_json, indent=2, sort_keys=True)

        # data info
        self.dataset.save_to_wandb(experiment)

    # ---------- Standardinzation ----------------
    def standardize_data(self):
        """Apply data normalization based on stats from training set"""
        self.dataset.standardize(self.training)

    # --------- Managing predictions on this data ---------
    def predict(self, model, save_to, sections=['test'], single_batch=False):
        """Save model predictions on the given dataset section"""
        # Main path
        prediction_path = save_to / ('nn_pred_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
        prediction_path.mkdir(parents=True, exist_ok=True)

        for section in sections:
            # Section path
            section_dir = prediction_path / section
            section_dir.mkdir(parents=True, exist_ok=True)

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()

            # turn on att weights saving during prediction!
            if hasattr(model, 'module'):
                model.module.save_att_weights = True
            else:
                model.save_att_weights = True   # model that don't have this poperty will just ignore it

            with torch.no_grad():
                loader = self.get_loader(section)
                if loader:
                    for batch in loader:
                        features_device = batch['features'].to(device)
                        preds = model(features_device)
                        self.dataset.save_prediction_batch(
                            preds, batch['name'], batch['data_folder'], section_dir, features=batch['features'].numpy(), 
                            model=model)
                        
                        if single_batch:  # stop after first iteration
                            break
            
            # Turn of to avoid wasting time\memory diring other operations
            if hasattr(model, 'module'):
                model.module.save_att_weights = False
            else:
                model.save_att_weights = False   # model that don't have this poperty will just ignore it

        return prediction_path


# ------------------ Transforms ----------------
def _dict_to_tensors(dict_obj):  # helper
    """convert a dictionary with numeric values into a new dictionary with torch tensors"""
    new_dict = dict.fromkeys(dict_obj.keys())
    for key, value in dict_obj.items():
        if value is None:
            new_dict[key] = torch.Tensor()
        elif isinstance(value, dict):
            new_dict[key] = _dict_to_tensors(value)
        elif isinstance(value, str):  # no changes for strings
            new_dict[key] = value
        elif isinstance(value, np.ndarray):
            new_dict[key] = torch.from_numpy(value)
            if value.dtype not in [np.int, np.bool]:
                new_dict[key] = new_dict[key].float()  # cast all doubles and ofther stuff to floats
        else:
            new_dict[key] = torch.tensor(value)  # just try directly, if nothing else works
    return new_dict


# Custom transforms -- to tensor
class SampleToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):        
        return _dict_to_tensors(sample)


class FeatureStandartization():
    """Normalize features of provided sample with given stats"""
    def __init__(self, shift, scale):
        self.shift = torch.Tensor(shift)
        self.scale = torch.Tensor(scale)
    
    def __call__(self, sample):
        updated_sample = {}
        for key, value in sample.items():
            if key == 'features':
                updated_sample[key] = (sample[key] - self.shift) / self.scale
            else: 
                updated_sample[key] = sample[key]

        return updated_sample


class GTtandartization():
    """Normalize features of provided sample with given stats
        * Supports multimodal gt represented as dictionary
        * For dictionary gts, only those values are updated for which the stats are provided
    """
    def __init__(self, shift, scale):
        """If ground truth is a dictionary in itself, the provided values should also be dictionaries"""
        
        self.shift = _dict_to_tensors(shift) if isinstance(shift, dict) else torch.Tensor(shift)
        self.scale = _dict_to_tensors(scale) if isinstance(scale, dict) else torch.Tensor(scale)
    
    def __call__(self, sample):
        gt = sample['ground_truth']
        if isinstance(gt, dict):
            new_gt = dict.fromkeys(gt.keys())
            for key, value in gt.items():
                new_gt[key] = value
                if key in self.shift:
                    new_gt[key] = new_gt[key] - self.shift[key]
                if key in self.scale:
                    new_gt[key] = new_gt[key] / self.scale[key]
                # if shift and scale are not set, the value is kept as it is
        else:
            new_gt = (gt - self.shift) / self.scale

        # gather sample
        updated_sample = {}
        for key, value in sample.items():
            updated_sample[key] = new_gt if key == 'ground_truth' else sample[key]

        return updated_sample


# --------------------- Datasets -------------------------

class BaseDataset(Dataset):
    """
        * Implements base interface for my datasets
        * Implements routines for datapoint retrieval, structure & cashing 
        (agnostic of the desired feature & GT structure representation)
    """
    def __init__(self, root_dir, start_config={'data_folders': []}, gt_caching=False, feature_caching=False, transforms=[]):
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
        self.transforms = [SampleToTensor()] + transforms

        # statistics already there --> need to apply it
        if 'standardize' in self.config:
            self.standardize()

        # DEBUG -- access all the datapoints to pre-populate the cache of the data
        # self._renew_cache()

        # in\out sizes
        self._estimate_data_shape()

    def save_to_wandb(self, experiment):
        """Save data cofiguration to current expetiment run"""
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

        # if datapoint_name == 'tee_AME9SSCR7X':
        #     print(self.transforms)
        #     print('After transform: {}'.format(sample['features']))

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
        ids_dict = dict.fromkeys(self.data_folders)
        index_list = np.array(index_list)
        
        # assign by comparing with data_folders start & end ids
        # enforce sort Just in case
        self.dataset_start_ids = sorted(self.dataset_start_ids, key=lambda idx: idx[1])

        for i in range(0, len(self.dataset_start_ids) - 1):
            ids_filter = (index_list >= self.dataset_start_ids[i][1]) & (index_list < self.dataset_start_ids[i + 1][1])
            ids_dict[self.dataset_start_ids[i][0]] = index_list[ids_filter]
        
        return ids_dict

    def subsets_per_datafolder(self, index_list=None):
        """
            Group given indices by datafolder and Return dictionary with Subset objects for each group.
            if None, a breakdown for the full dataset is given
        """
        if index_list is None:
            index_list = range(len(self))
        per_data = self.indices_by_data_folder(index_list)
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
            folder_elements = [file.name for file in (self.root_path / datapoint_name).glob('*')]  # all files in this directory
            ground_truth = self._get_ground_truth(datapoint_name, folder_elements)
            if self.gt_caching:
                self.gt_cached[datapoint_name] = ground_truth
        
        if datapoint_name in self.feature_cached:
            features = self.feature_cached[datapoint_name]
        else:
            folder_elements = folder_elements if folder_elements is not None else [file.name for file in (self.root_path / datapoint_name).glob('*')]
            features = self._get_features(datapoint_name, folder_elements)
            
            if self.feature_caching:  # save read values 
                self.feature_cached[datapoint_name] = features
        
        return features, ground_truth

    def _get_features(self, datapoint_name, folder_elements=None):
        """Read/generate datapoint features"""
        return np.array([0])

    def _get_ground_truth(self, datapoint_name, folder_elements=None):
        """Ground thruth prediction for a datapoint"""
        return np.array([0])

    def _estimate_data_shape(self):
        """Get sizes/shapes of a datapoint for external references"""
        elem = self[0]
        feature_size = elem['features'].shape[0]
        gt_size = elem['ground_truth'].shape[0] if hasattr(elem['ground_truth'], 'shape') else None
        # sanity checks
        if ('feature_size' in self.config and feature_size != self.config['feature_size'] 
                or 'ground_truth_size' in self.config and gt_size != self.config['ground_truth_size']):
            raise RuntimeError('BaseDataset::Error::feature shape ({}) or ground truth shape ({}) from loaded config do not match calculated values: {}, {}'.format(
                self.config['feature_size'], self.config['ground_truth_size'], feature_size, gt_size))

        self.config['feature_size'], self.config['ground_truth_size'] = feature_size, gt_size

    def _update_on_config_change(self):
        """Update object inner state after config values have changed"""
        pass


class GarmentBaseDataset(BaseDataset):
    """Base class to work with data from custom garment datasets"""
        
    def __init__(self, root_dir, start_config={'data_folders': []}, gt_caching=False, feature_caching=False, transforms=[]):
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

        super().__init__(root_dir, start_config, gt_caching=gt_caching, feature_caching=feature_caching, transforms=transforms)

        # To make sure the datafolder names are unique after updates
        all_nicks = self.data_folders_nicknames.values()
        if len(all_nicks) > len(set(all_nicks)):
            print('{}::Warning::Some data folder nicknames are not unique: {}. Reverting to the use of original folder names'.format(
                self.__class__.__name__, self.data_folders_nicknames
            ))
            self.data_folders_nicknames = dict(zip(self.data_folders, self.data_folders))

        # evaluate base max values for number of panels, number of edges in panels among pattern in all the datasets
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

        for dataset_folder in self.data_folders:
            shutil.copy(
                self.root_path / dataset_folder / 'dataset_properties.json', 
                experiment.local_path() / (dataset_folder + '_properties.json'))
    
    # ------ Garment Data-specific basic functions --------
    def _clean_datapoint_list(self, datapoints_names, dataset_folder):
        """
            Remove all elements marked as failure from the provided list
            Updates the currect dataset nickname as a small sideeffect
        """

        dataset_props = Properties(self.root_path / dataset_folder / 'dataset_properties.json')
        if not dataset_props['to_subfolders']:
            raise NotImplementedError('Only working with datasets organized with subfolders')

        # NOTE A little side-effect here, since we are loading the dataset_properties anyway
        self.data_folders_nicknames[dataset_folder] = dataset_props['templates'].split('/')[-1].split('.')[0]

        try: 
            datapoints_names.remove(dataset_folder + '/renders')  # TODO read ignore list from props
            print('Dataset {}:: /renders/ subfolder ignored'.format(dataset_folder))
        except ValueError:  # it's ok if there is no subfolder for renders
            print('GarmentBaseDataset::Info::No renders subfolder found in {}'.format(dataset_folder))
            pass

        fails_dict = dataset_props['sim']['stats']['fails']
        # TODO allow not to ignore some of the subsections
        for subsection in fails_dict:
            for fail in fails_dict[subsection]:
                try:
                    datapoints_names.remove(dataset_folder + '/' + fail)
                    print('Dataset {}:: {} ignored'.format(dataset_folder, fail))
                except ValueError:  # if fail was already removed based on previous failure subsection
                    pass
        
        return datapoints_names

    # ------------- Datapoints Utils --------------
    def _sample_points(self, datapoint_name, folder_elements):
        """Make a sample from the 3d surface from a given datapoint files"""
        obj_list = [file for file in folder_elements if self.config['obj_filetag'] in file and '.obj' in file]
        if not obj_list:
            raise RuntimeError('Dataset:Error: geometry file *{}*.obj not found for {}'.format(self.config['obj_filetag'], datapoint_name))
        
        verts, faces = igl.read_triangle_mesh(str(self.root_path / datapoint_name / obj_list[0]))
        points = GarmentBaseDataset.sample_mesh_points(self.config['mesh_samples'], verts, faces)

        # Debug
        # if 'skirt_4_panels_00HUVRGNCG' in datapoint_name:
        #     meshplot.offline()
        #     meshplot.plot(points, c=points[:, 0], shading={"point_size": 3.0})
        return points

    def _read_pattern(self, datapoint_name, folder_elements, 
                      pad_panels_to_len=None, pad_panel_num=None, pad_stitches_num=None,
                      with_placement=False, with_stitches=False, with_stitch_tags=False):
        """Read given pattern in tensor representation from file"""
        spec_list = [file for file in folder_elements if 'specification.json' in file]
        if not spec_list:
            raise RuntimeError('GarmentBaseDataset::Error::*specification.json not found for {}'.format(datapoint_name))
        
        pattern = NNSewingPattern(self.root_path / datapoint_name / spec_list[0])
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

        pattern = NNSewingPattern(view_ids=False)
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
    def __init__(self, root_dir, start_config={'data_folders': []}, gt_caching=False, feature_caching=False, transforms=[]):
        if 'mesh_samples' not in start_config:
            start_config['mesh_samples'] = 2000  # default value if not given -- a bettern gurantee than a default value in func params
        super().__init__(root_dir, start_config, 
                         gt_caching=gt_caching, feature_caching=feature_caching, transforms=transforms)
        
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
        self.transforms = [transform for transform in self.transforms if not isinstance(transform, GTtandartization) and not isinstance(transform, FeatureStandartization)]

        self.transforms.append(GTtandartization(stats['gt_shift'], stats['gt_scale']))
        self.transforms.append(FeatureStandartization(stats['f_shift'], stats['f_scale']))

    def save_prediction_batch(self, predictions, datanames, data_folders, save_to, features=None, weights=None, **kwargs):
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

            pattern = self._pred_to_pattern(prediction, name)

            # save prediction
            folder_nick = self.data_folders_nicknames[folder]

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

    @staticmethod
    def empty_panels_mask(num_panels, tot_length):
        """Empty panels as boolean mask"""

        mask = np.zeros(tot_length, dtype=np.bool)
        mask[num_panels:] = True

        return mask

    def _get_features(self, datapoint_name, folder_elements):
        """Get mesh vertices for given datapoint with given file list of datapoint subfolder"""
        points = self._sample_points(datapoint_name, folder_elements)
        return points  # return in 3D
      
    def _get_ground_truth(self, datapoint_name, folder_elements):
        """Get the pattern representation with 3D placement"""
        pattern, num_edges, num_panels, rots, tranls, stitches, num_stitches, stitch_tags = self._read_pattern(
            datapoint_name, folder_elements, 
            pad_panels_to_len=self.config['max_panel_len'],
            pad_panel_num=self.config['max_pattern_len'],
            pad_stitches_num=self.config['max_num_stitches'],
            with_placement=True, with_stitches=True, with_stitch_tags=True)
        free_edges_mask = self.free_edges_mask(pattern, stitches, num_stitches)
        empty_panels_mask = self.empty_panels_mask(num_panels, len(pattern))  # useful for evaluation

        return {
            'outlines': pattern, 'num_edges': num_edges,
            'rotations': rots, 'translations': tranls, 
            'num_panels': num_panels, 'empty_panels_mask': empty_panels_mask, 'num_stitches': num_stitches,
            'stitches': stitches, 'free_edges_mask': free_edges_mask, 'stitch_tags': stitch_tags}

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

        # stitch tags to stitch list
        stitches = self.tags_to_stitches(
            torch.from_numpy(prediction['stitch_tags']) if isinstance(prediction['stitch_tags'], np.ndarray) else prediction['stitch_tags'],
            prediction['free_edges_mask']
        )

        return self._pattern_from_tenzor(
            dataname, 
            prediction['outlines'], prediction['rotations'], prediction['translations'], 
            stitches, 
            std_config={}, supress_error=True)


class Garment2DPatternDataset(Garment3DPatternFullDataset):
    """Dataset definition for 2D pattern autoencoder
        * features: a 'front' panel edges represented as a sequence
        * ground_truth is not used as in Panel dataset"""
    def __init__(self, root_dir, start_config={'data_folders': []}, gt_caching=False, feature_caching=False, transforms=[]):
        super().__init__(root_dir, start_config, gt_caching=gt_caching, feature_caching=feature_caching, transforms=transforms)
    
    def _get_features(self, datapoint_name, folder_elements):
        """Get mesh vertices for given datapoint with given file list of datapoint subfolder"""
        return self._get_ground_truth(datapoint_name, folder_elements)['outlines']
        
    def _pred_to_pattern(self, prediction, dataname):
        """Convert given predicted value to pattern object"""
        prediction = prediction['outlines'].cpu().numpy()

        gt_shifts = self.config['standardize']['gt_shift']['outlines']
        gt_scales = self.config['standardize']['gt_scale']['outlines']
        prediction = prediction * gt_scales + gt_shifts

        return self._pattern_from_tenzor(
            dataname, 
            prediction, 
            std_config={}, 
            supress_error=True)


class GarmentStitchPairsDataset(GarmentBaseDataset):
    """
        Dataset targets the task of predicting if a particular pair of edges is connected by a stitch or not
    """
    def __init__(self, root_dir, start_config={'data_folders': []}, gt_caching=False, feature_caching=False, transforms=[]):
        if gt_caching or feature_caching:
            gt_caching = feature_caching = True  # ensure that both are simulataneously True or False
        
        # data-specific defaults
        init_config = {
            'data_folders': [],
            'edge_pairs_num': 100,
            'shuffle_pairs': False, 
            'shuffle_pairs_order': False
        }
        init_config.update(start_config)  # values from input

        super().__init__(root_dir, init_config, 
                         gt_caching=gt_caching, feature_caching=feature_caching, transforms=transforms)

        self.config.update(
            element_size=self[0]['features'].shape[-1],
        )
        

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
                # TODO decide on the type of stats needed
                feature_shift, feature_scale = self._get_distribution_stats(batch['features'], padded=False)
                break  # only one batch out there anyway

            self.config['standardize'] = {
                'f_shift': feature_shift.cpu().numpy(), 
                'f_scale': feature_scale.cpu().numpy(),
            }
            stats = self.config['standardize']
        else:  # nothing is provided
            raise ValueError('Garment3DPatternFullDataset::Error::Standardization cannot be applied: supply either stats in config or training set to use standardization')

        # clean-up tranform list to avoid duplicates
        self.transforms = [transform for transform in self.transforms if not isinstance(transform, GTtandartization) and not isinstance(transform, FeatureStandartization)]

        self.transforms.append(FeatureStandartization(stats['f_shift'], stats['f_scale']))


    def save_prediction_batch(self, predictions, datanames, data_folders, save_to, model=None, **kwargs):
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
            folder_nick = self.data_folders_nicknames[folder]
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
        
        pattern = NNSewingPattern(self.root_path / datapoint_name / spec_list[0])
        features, ground_truth = pattern.stitches_as_3D_pairs(self.config['edge_pairs_num'], self.config['shuffle_pairs_order'])
        
        # save elements
        if self.gt_caching and self.feature_caching:
            self.gt_cached[datapoint_name] = ground_truth
            self.feature_cached[datapoint_name] = features
        
        return features, ground_truth


# ------------------------- Utils for non-dataset examples --------------------------

def sample_points_from_meshes(mesh_paths, data_config):
    """
        Sample points from the given list of triangle meshes (as .obj files -- or other file formats supported by libigl)
    """
    points_list = []
    for mesh in mesh_paths:
        verts, faces = igl.read_triangle_mesh(str(mesh))
        points = GarmentBaseDataset.sample_mesh_points(data_config['mesh_samples'], verts, faces)
        if 'standardize' in data_config:
            points = (points - data_config['standardize']['f_shift']) / data_config['standardize']['f_scale']
        points_list.append(torch.Tensor(points))
    return points_list


def save_garments_prediction(predictions, save_to, data_config=None, datanames=None):
    """ 
        Saving arbitrary sewing pattern predictions that
        
        * They do NOT have to be coming from garmet dataset samples.
    """

    save_to = Path(save_to)
    batch_size = predictions['outlines'].shape[0]

    if datanames is None:
        datanames = ['pred_{}'.format(i) for i in range(batch_size)]
        
    for idx, name in enumerate(datanames):
        # "unbatch" dictionary
        prediction = {}
        for key in predictions:
            prediction[key] = predictions[key][idx]

        if data_config is not None and 'standardize' in data_config:
            # undo standardization  (outside of generinc conversion function due to custom std structure)
            gt_shifts = data_config['standardize']['gt_shift']
            gt_scales = data_config['standardize']['gt_scale']
            for key in gt_shifts:
                if key == 'stitch_tags' and not data_config['explicit_stitch_tags']:  
                    # ignore stitch tags update if explicit tags were not used
                    continue
                prediction[key] = prediction[key].cpu().numpy() * gt_scales[key] + gt_shifts[key]

        # stitch tags to stitch list
        stitches = Garment3DPatternFullDataset.tags_to_stitches(
            torch.from_numpy(prediction['stitch_tags']) if isinstance(prediction['stitch_tags'], np.ndarray) else prediction['stitch_tags'],
            prediction['free_edges_mask']
        )

        pattern = VisPattern(view_ids=False)
        pattern.name = name
        try:
            pattern.pattern_from_tensors(
                prediction['outlines'], prediction['rotations'], prediction['translations'], 
                stitches=stitches,
                padded=True)   
            # save
            pattern.serialize(save_to, to_subfolder=True)
        except (RuntimeError, InvalidPatternDefError, TypeError) as e:
            print(e)
            print('Saving predictions::Skipping pattern {}'.format(name))
            pass


if __name__ == "__main__":

    # data_location = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
    system = Properties('./system.json')
    # dataset_folder = 'data_1000_skirt_4_panels_200616-14-14-40'
    dataset_folder = 'data_1000_tee_200527-14-50-42_regen_200612-16-56-43'

    data_location = Path(system['datasets_path']) / dataset_folder

    dataset = Garment3DPatternFullDataset(system['datasets_path'], {
        'data_folders': [
            'data_1000_tee_200527-14-50-42_regen_200612-16-56-43',
            'data_1000_skirt_4_panels_200616-14-14-40'
        ]
    })

    print(len(dataset), dataset.config)
    # print(dataset[0]['name'], dataset[0]['data_folder'])
    # print(dataset[0]['ground_truth'])
    # print(dataset[-1]['name'], dataset[-1]['data_folder'])
    # print(dataset[-1]['ground_truth'])

    # print(dataset[5]['ground_truth'])

    datawrapper = DatasetWrapper(dataset)
    datawrapper.new_split(10, 10, 3000)

    # datawrapper.standardize_data()

    # print(dataset.config['standardize'])

    # print(dataset[0]['ground_truth'])
    # print(dataset[5]['features'])

    # stitch_tags = torch.Tensor(
    #     [[
    #         [-5.20318419e+01,  2.28511632e+01, -2.51693441e-01],
    #         [-3.37806229e+01,  1.83484274e+01, -1.34557098e+01],
    #         [-4.78298848e+01, -3.23568072e+00,  5.23204596e-01],
    #         [-2.24093100e+00,  2.60038064e+00, -3.99605272e-01],
    #         [-2.63538333e+00, -1.33136284e+00, -2.10666308e-01],
    #         [-1.63031343e+00, -2.54933174e-01, -3.51316654e-01],
    #         [-1.39121696e+00, -3.99988596e-01, -2.81176007e-01],
    #         [-2.30340490e+00, -8.96057867e-01, -2.23693146e-01],
    #         [-1.34838584e+00, -7.02625742e-01, -1.68379548e-01]],

    #         # [[-3.76714804e+01, -7.83406514e-01,  4.21872022e+00],
    #         # [-3.36618020e+01, 2.02843384e+01,  1.38874859e+01],
    #         # [-2.98077958e+01, 1.76473066e+01, -2.72662691e-01],
    #         # [-1.16667320e+01, -2.32238589e-01, -2.70201479e-01],
    #         # [-4.62600892e+00, -1.03453005e+00, -3.87780018e-01],
    #         # [-1.12707816e+00, -9.99629252e-01, -4.63383672e-01],
    #         # [-1.36449657e+00, -1.40015080e+00, -3.52761674e-01],
    #         # [-2.49496085e+00, -9.37499225e-01,  1.75160368e-02],
    #         # [-1.82691709e+00, -9.67022458e-01, -2.44112010e-02]],

    #     # [[-3.08226939e+01, -4.40332624e+01,  5.20345150e-02],
    #     # [-2.57602088e+01,  9.90067487e+00, -1.46536128e+01],
    #     # [-1.71379921e+01,  2.79928684e+01,  1.99508048e+00],
    #     # [-2.83926761e+00, -1.74896668e+00, -5.10048808e-01],
    #     # [ 1.69106852e+01,  2.67701350e+01,  1.92687865e+00],
    #     # [ 2.33531630e+01,  1.87797418e+01, -1.42510374e+01],
    #     # [ 2.93172584e+01, -4.47819890e+01,  3.36183070e-02],
    #     # [ 1.33929771e+00, -7.66815033e-01,  3.39337064e-01],
    #     # [-2.58251768e+00, -3.01449654e+00, -8.38604599e-01]],

    #     # [[ 7.60947669e+00, -2.93252076e+00,  1.52037176e-01],
    #     # [ 3.81844651e+01, -4.00190576e+01,  3.03353006e+00],
    #     # [ 2.92371784e+01,  1.43307176e+01,  1.46076412e+01],
    #     # [ 1.18328286e+01,  2.77603316e+01,  1.56419596e+00],
    #     # [-2.90668095e+00,  8.09845111e-01,  2.66955523e-02],
    #     # [-2.56676557e+00,  4.74590501e-01, -5.36622275e-01],
    #     # [-1.68190322e+01,  2.80625313e+01,  2.39942965e+00],
    #     # [-2.54365294e+01,  1.39684045e+01,  1.59379294e+01],
    #     # [-2.91702785e+01, -4.18118565e+01,  9.33112066e-01]],

    #     # [[ 4.14145748e+01,  3.80079295e+00, -5.11258303e+00],
    #     # [ 3.60885075e+01, 2.16267973e+01, -9.39987246e+00],
    #     # [ 3.18748450e+01,  2.12366894e+01,  4.64202212e-01],
    #     # [ 1.24644558e+01,  1.05895206e+00,  5.83006592e-01],
    #     # [ 4.96519200e-01,  1.38524990e-01,  1.54455938e-01],
    #     # [-1.03008224e+00, -6.21098086e-01, -1.10430334e-02],
    #     # [-1.33714690e+00, -5.25571702e-01, -1.00064235e-01],
    #     # [-1.87922790e+00, -7.71050929e-01, -2.45438618e-01],
    #     # [-1.45271650e+00, -4.56986468e-01, -2.65114454e-01]],

    #     # [[ 3.84707164e+01,  2.86529960e+01,  8.99072663e-01],
    #     # [ 3.08633876e+01,  1.25824877e+01,  1.59490529e+01],
    #     # [ 4.77875079e+01, -4.37732398e+00,  1.27716544e+00],
    #     # [ 1.48898335e+00, -4.72142368e-02, -4.45565817e-02],
    #     # [-7.72547412e-01, -1.78853016e+00,  1.03305003e-01],
    #     # [-1.33450125e+00, -1.22900781e+00, -4.36989260e-02],
    #     # [-1.32287662e+00, -2.99308717e-01, -1.39927901e-01],
    #     # [-1.73496211e+00, -4.85474734e-01, -1.33751047e-01],
    #     # [-8.70132007e-01, -2.93109585e-01, -2.64518426e-01]]
    #     ])

    # print(Garment3DPatternFullDataset.tags_to_stitches(stitch_tags, torch.full((stitch_tags.shape[0], stitch_tags.shape[1]), 0.)))
