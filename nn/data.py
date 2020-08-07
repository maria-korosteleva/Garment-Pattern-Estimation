import numpy as np
import os
from pathlib import Path
import shutil
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import igl
# import meshplot  # when uncommented, could lead to problems with wandb run syncing

# My modules
from customconfig import Properties
from pattern.core import ParametrizedPattern, BasicPattern
from pattern.wrappers import VisPattern

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

        self.batch_size = None
        self.loader_full = None
        self.loader_train = None
        self.loader_validation = None
        self.loader_test = None

        self.split_info = {
            'random_seed': None, 
            'valid_percent': None, 
            'test_percent': None
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
        elif data_section == 'train':
            return self.loader_train
        elif data_section == 'test':
            return self.loader_test
        elif data_section == 'validation':
            return self.loader_validation
        
        raise ValueError('DataWrapper::requested loader on unknown data section {}'.format(data_section))

    def new_loaders(self, batch_size=None, shuffle_train=True):
        """Create loaders for current data split. Note that result depends on the random number generator!"""
        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is None:
            raise RuntimeError('DataWrapper:Error:cannot create loaders: batch_size is not set')

        self.loader_train = DataLoader(self.training, self.batch_size, shuffle=shuffle_train)
        self.loader_validation = DataLoader(self.validation, self.batch_size) if self.validation else None
        self.loader_test = DataLoader(self.test, self.batch_size) if self.test else None
        self.loader_full = DataLoader(self.dataset, self.batch_size)

        return self.loader_train, self.loader_validation, self.loader_test

    # -------- Reproducibility ---------------
    def new_split(self, valid_percent, test_percent=None, random_seed=None):
        """Creates train/validation or train/validation/test splits
            depending on provided parameters
            """
        self.split_info['random_seed'] = random_seed if random_seed else int(time.time())
        self.split_info.update(valid_percent=valid_percent, test_percent=test_percent)
        
        return self.load_split()

    def load_split(self, split_info=None, batch_size=None):
        """Get the split by provided parameters. Can be used to reproduce splits on the same dataset.
            NOTE this function re-initializes torch random number generator!
        """
        if split_info:
            if any([key not in split_info for key in self.split_info]):
                raise ValueError('Specified split information is not full: {}'.format(split_info))
            self.split_info = split_info

        torch.manual_seed(self.split_info['random_seed'])
        valid_size = (int) (len(self.dataset) * self.split_info['valid_percent'] / 100)
        if self.split_info['test_percent']:
            test_size = (int) (len(self.dataset) * self.split_info['test_percent'] / 100)
            self.training, self.validation, self.test = torch.utils.data.random_split(
                self.dataset, (len(self.dataset) - valid_size - test_size, valid_size, test_size))
        else:
            self.training, self.validation = torch.utils.data.random_split(
                self.dataset, (len(self.dataset) - valid_size, valid_size))
            self.test = None

        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is not None:
            self.new_loaders()  # s.t. loaders could be used right away

        print ('{} split: {} / {} / {}'.format(self.dataset.name, len(self.training), 
                                               len(self.validation) if self.validation else None, 
                                               len(self.test) if self.test else None))

        return self.training, self.validation, self.test

    def save_to_wandb(self, experiment):
        """Save current data info to the wandb experiment"""
        # Split
        experiment.add_config('data_split', self.split_info)
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
        prediction_path = save_to / ('nn_pred_' + self.dataset.name + datetime.now().strftime('%y%m%d-%H-%M-%S'))
        prediction_path.mkdir(parents=True, exist_ok=True)

        for section in sections:
            # Section path
            section_dir = prediction_path / section
            section_dir.mkdir(parents=True, exist_ok=True)

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            with torch.no_grad():
                loader = self.get_loader(section)
                if loader:
                    if single_batch:
                        batch = next(iter(loader))    # might have some issues, see https://github.com/pytorch/pytorch/issues/1917
                        features = batch['features'].to(device)
                        self.dataset.save_prediction_batch(model(features), batch['name'], section_dir)
                    else:
                        for batch in loader:
                            features = batch['features'].to(device)
                            self.dataset.save_prediction_batch(model(features), batch['name'], section_dir)
        return prediction_path

# ------------------ Transforms ----------------
# Custom transforms -- to tensor
class SampleToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        features, gt = sample['features'], sample['ground_truth']
        
        return {
            'features': torch.from_numpy(features).float() if features is not None else torch.Tensor(), 
            'ground_truth': torch.from_numpy(gt).float() if gt is not None else torch.Tensor(), 
            'name': sample['name']
        }

class FeatureStandartizatoin():
    """Normalize features of provided sample with given stats"""
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    
    def __call__(self, sample):
        return {
            'features': (sample['features'] - self.mean) / self.std,
            'ground_truth': sample['ground_truth'],
            'name': sample['name']
        }

class GTtandartizatoin():
    """Normalize features of provided sample with given stats"""
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    
    def __call__(self, sample):
        return {
            'features': sample['features'],
            'ground_truth': (sample['ground_truth'] - self.mean) / self.std,
            'name': sample['name']
        }

# --------------------- Datasets -------------------------

class BaseDataset(Dataset):
    """Ensure that all my datasets follow this interface"""
    def __init__(self, root_dir, start_config={}, gt_caching=False, feature_caching=False, transforms=[]):
        """Kind of Universal init for my datasets
            if cashing is enabled, datapoints will stay stored in memory on first call to them: might speed up data processing by reducing file reads"""
        self.root_path = Path(root_dir)
        self.name = self.root_path.name
        self.config = {}
        self.update_config(start_config)
        
        # list of items = subfolders
        _, dirs, _ = next(os.walk(self.root_path))
        self.datapoints_names = dirs
        self._clean_datapoint_list()
        self.cached = {}
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

        # in\out sizes
        self._estimate_data_shape()

        # statistics already there
        if 'standardize' in self.config:
            self.standardize(stats=self.config['standardize'])

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
        
        sample = {'features': features, 'ground_truth': ground_truth, 'name': datapoint_name}

        # apply transfomations (equally to samples from files or from cache)
        for transform in self.transforms:
            sample = transform(sample)

        # if datapoint_name == 'tee_0BEJ3JZP2O':
        #    print(sample)

        return sample

    def update_config(self, in_config):
        """Define dataset configuration:
            * to be part of experimental setup on wandb
            * Control obtainign values for datapoints"""
        self.config.update(in_config)
        if 'name' in self.config and self.name != self.config['name']:
            print('BaseDataset:Warning:dataset name ({}) in loaded config does not match current dataset name ({})'.format(self.config['name'], self.name))

        self.config['name'] = self.name
        self._update_on_config_change()

    # -------- Data-specific functions --------
    def save_prediction_batch(self, predictions, datanames, save_to):
        """Saves predicted params of the datapoint to the original data folder"""
        pass

    def standardize(self, training=None, stats={}):
        """Use element normalization\standardization based on stats from the training subset.
            Dataset is the object most aware of the datapoint structure hence it's the place to calculate & use the normalization.
            Accepts either of two inputs: 
            * training subset to calculate the data statistics -- the stats are only based on training subsection of the data
            * if stats info is given, it's used instead of calculating new statistics (usually when calling to restore dataset from existing experiment)
            'training' parameter has a priority: if it's given, the statistics are recalculated despte the value in 'stats' param
            """
        print('{}::Warning::No normalization is implemented'.format(self.__class__.__name__))

    def _clean_datapoint_list(self):
        """Remove non-datapoints subfolders, failing cases, etc. Children are to override this function when needed"""
        # See https://stackoverflow.com/questions/57042695/calling-super-init-gives-the-wrong-method-when-it-is-overridden
        pass

    def _get_features(self, datapoint_name, folder_elements=None):
        """Read/generate datapoint features"""
        return np.array([0])

    def _get_ground_truth(self, datapoint_name, folder_elements=None):
        """Ground thruth prediction for a datapoint"""
        return np.array([0])

    def _estimate_data_shape(self):
        """Get sizes/shapes of a datapoint for external references"""
        elem = self[0]
        feature_size, gt_size = elem['features'].shape[0], elem['ground_truth'].shape[0]
        # sanity checks
        if ('feature_size' in self.config and feature_size != self.config['feature_size']
                or 'ground_truth_size' in self.config and gt_size != self.config['ground_truth_size']):
            raise RuntimeError('BaseDataset::Error::feature shape ({}) or ground truth shape ({}) from loaded config do not match calculated values: {}, {}'.format(
                self.config['feature_size'],  self.config['ground_truth_size'], feature_size, gt_size))

        self.config['feature_size'], self.config['ground_truth_size'] = feature_size, gt_size

    def _update_on_config_change(self):
        """Update object inner state after config values have changed"""
        pass


class GarmentBaseDataset(BaseDataset):
    """Base class to work with data from custom garment datasets"""
        
    def __init__(self, root_dir, start_config={}, gt_caching=False, feature_caching=False, transforms=[]):
        """
        Args:
            root_dir (string): Directory with all examples as subfolders
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_props = Properties(Path(root_dir) / 'dataset_properties.json')
        if not self.dataset_props['to_subfolders']:
            raise NotImplementedError('Working with datasets with all datapopints ')
        
        super().__init__(root_dir, start_config, gt_caching=gt_caching, feature_caching=feature_caching, transforms=transforms)
     
    def save_to_wandb(self, experiment):
        """Save data cofiguration to current expetiment run"""
        super().save_to_wandb(experiment)

        shutil.copy(self.root_path / 'dataset_properties.json', experiment.local_path())
    
    def save_prediction_batch(self, predictions, datanames, save_to):
        """Saves predicted params of the datapoint to the original data folder.
            Returns list of paths to files with prediction visualizations"""

        prediction_imgs = []
        for prediction, name in zip(predictions, datanames):

            pattern = self._pred_to_pattern(prediction, name)

            # save
            final_dir = pattern.serialize(save_to, to_subfolder=True, tag='_predicted_')
            final_file = pattern.name + '_predicted__pattern.png'
            prediction_imgs.append(Path(final_dir) / final_file)

            # copy originals for comparison
            for file in (self.root_path / name).glob('*'):
                if ('.png' in file.suffix) or ('.json' in file.suffix):
                    shutil.copy2(str(file), str(final_dir))
        return prediction_imgs

    # ------ Garment Data-specific basic functions --------
    def _clean_datapoint_list(self):
        """Remove all elements marked as failure from the datapoint list"""
        self.datapoints_names.remove('renders')  # TODO read ignore list from props

        fails_dict = self.dataset_props['sim']['stats']['fails']
        # TODO allow not to ignore some of the subsections
        for subsection in fails_dict:
            for fail in fails_dict[subsection]:
                try:
                    self.datapoints_names.remove(fail)
                    print('Dataset:: {} ignored'.format(fail))
                except ValueError:  # if fail was already removed based on previous failure subsection
                    pass

    def _pred_to_pattern(self, prediction, dataname):
        """Convert given predicted value to pattern object"""
        return None

    # ------------- Datapoints Utils --------------
    def _sample_points(self, datapoint_name, folder_elements):
        """Make a sample from the 3d surface of a given datapoint"""
        obj_list = [file for file in folder_elements if 'sim.obj' in file]
        if not obj_list:
            raise RuntimeError('Dataset:Error: geometry file *sim.obj not found for {}'.format(datapoint_name))
        
        verts, faces = igl.read_triangle_mesh(str(self.root_path / datapoint_name / obj_list[0]))

        num_samples = self.config['mesh_samples']

        barycentric_samples, face_ids = igl.random_points_on_mesh(num_samples, verts, faces)
        face_ids[face_ids >= len(faces)] = len(faces) - 1  # workaround for https://github.com/libigl/libigl/issues/1531

        # convert to normal coordinates
        points = np.empty(barycentric_samples.shape)
        for i in range(len(face_ids)):
            face = faces[face_ids[i]]
            barycentric_coords = barycentric_samples[i]
            face_verts = verts[face]
            points[i] = np.dot(barycentric_coords, face_verts)

        # Debug
        # if datapoint_name == 'skirt_4_panels_00HUVRGNCG':
        #     meshplot.offline()
        #     meshplot.plot(points, c=points[:, 0], shading={"point_size": 3.0})
        return points

    def _read_pattern(self, datapoint_name, folder_elements):
        """Read given pattern in tensor representation from file"""
        spec_list = [file for file in folder_elements if 'specification.json' in file]
        if not spec_list:
            raise RuntimeError('GarmentBaseDataset::Error::*specification.json not found for {}'.format(datapoint_name))
        
        pattern = BasicPattern(self.root_path / datapoint_name / spec_list[0])
        return pattern.pattern_as_tensor()

    def _pattern_from_tenzor(self, dataname, tenzor, std_config={}, supress_error=True):
        """Shortcut to create a pattern object from given tenzor and suppress exceptions if those arize"""
        if std_config and 'standardize' in std_config:
            tenzor = tenzor * self.config['standardize']['std'] + self.config['standardize']['mean']

        pattern = VisPattern(view_ids=False)
        pattern.name = dataname
        try: 
            pattern.pattern_from_tensor(tenzor, padded=True)   
        except RuntimeError as e:
            if not supress_error:
                raise e
            print('Garment3DPatternDataset::Warning::{}: {}'.format(dataname, e))
            pass

        return pattern

    # -------- Generalized Utils
    def _unpad(self, element, tolerance=1.e-5):
        """Return copy of input element without padding from given element. Used to unpad edge sequences in pattern-oriented datasets"""
        # NOTE: might be some false removal of zero edges in the middle of the list.
        if torch.is_tensor(element):        
            bool_matrix = torch.isclose(element, torch.zeros_like(element), atol=tolerance)  # per-element comparison with zero
            selection = ~torch.all(bool_matrix, axis=1)  # only non-zero rows
        else:  # numpy
            selection = ~np.all(np.isclose(element, 0, atol=tolerance), axis=1)  # only non-zero rows
        return element[selection]

    def _get_stats(self, input_batch, padded=False):
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


class GarmentParamsDataset(GarmentBaseDataset):
    """
    For loading the custom generated data:
        * features: Coordinates of 3D mesh sample points flattened into vector
        * Ground_truth: parameters used to generate a garment
    """
    
    def __init__(self, root_dir, start_config={'mesh_samples': 1000}, gt_caching=False, feature_caching=False, transforms=[]):
        """
        Args:
            root_dir (string): Directory with all examples as subfolders
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if 'mesh_samples' not in start_config:  
            start_config['mesh_samples'] = 1000  # some default to ensure it's set

        super().__init__(root_dir, start_config, gt_caching=gt_caching, feature_caching=feature_caching, transforms=transforms)
    
    # ------ Data-specific basic functions --------
    def _get_features(self, datapoint_name, folder_elements):
        """Get mesh vertices for given datapoint with given file list of datapoint subfolder"""
        points = self._sample_points(datapoint_name, folder_elements)

        return points.ravel()  # flat vector as a feature
        
    def _get_ground_truth(self, datapoint_name, folder_elements):
        """Pattern parameters from a given datapoint subfolder"""
        spec_list = [file for file in folder_elements if 'specification.json' in file]
        if not spec_list:
            raise RuntimeError('Dataset:Error: *specification.json not found for {}'.format(datapoint_name))
        
        pattern = ParametrizedPattern(self.root_path / datapoint_name / spec_list[0])

        return np.array(pattern.param_values_list())
   
    def _pred_to_pattern(self, prediction, dataname):
        """Convert given predicted value to pattern object"""
        prediction = prediction.tolist()
        pattern = VisPattern(str(self.root_path / dataname / 'specification.json'), view_ids=False)  # with correct pattern name

        # apply new parameters
        pattern.apply_param_list(prediction)

        return pattern


class Garment3DParamsDataset(GarmentParamsDataset):
    """For loading the custom generated data:
        * features: list of 3D coordinates of 3D mesh sample points (2D matrix)
        * Ground_truth: parameters used to generate a garment
    """
    def __init__(self, root_dir, start_config={'mesh_samples': 1000}, gt_caching=False, feature_caching=False, transforms=[]):
        super().__init__(root_dir, start_config, gt_caching=gt_caching, feature_caching=feature_caching, transforms=transforms)
    
    # the only difference with parent class in the shape of the features
    def _get_features(self, datapoint_name, folder_elements):
        points = self._sample_points(datapoint_name, folder_elements)
        return points  # return in 3D


class GarmentPanelDataset(GarmentBaseDataset):
    """Experimental loading of custom generated data to be used in AE:
        * features: a panel edges represented as a sequence. Panel is chosen randomly. 
        * ground_truth is not used
        * When saving predictions, the predicted panel is always saved as panel with name provided in config 

    """
    def __init__(self, root_dir, start_config={'panel_name': 'front'}, gt_caching=False, feature_caching=False, transforms=[]):
        super().__init__(root_dir, start_config, gt_caching=gt_caching, feature_caching=feature_caching, transforms=transforms)
        self.config['element_size'] = self[0]['features'].shape[1]
    
    def standardize(self, training=None, stats={}):
        """Use mean&std for normalization of output features & restoring input predictions.
            Accepts either of two inputs: 
            * training subset to calculate the data statistics -- the stats are only based on training subsection of the data
            * if stats info is given, it's used instead of calculating new statistics (usually when calling to restore dataset from existing experiment)
            'training' parameter has a priority: if it's given, the statistics are recalculated despte the value in 'stats' param
        """
        print('GarmentPanelDataset::Using data normalization')

        if training is not None:
            loader = DataLoader(training, batch_size=len(training), shuffle=False)
            for batch in loader:
                feature_mean, feature_stds = self._get_stats(batch['features'], padded=True)
                # only one batch out there anyway
                break
            self.config['standardize'] = {'mean' : feature_mean.cpu().numpy(), 'std': feature_stds.cpu().numpy()}
            stats = self.config['standardize']
        elif not stats:
            raise ValueError('GarmentPanelDataset::Error::Standardization cannot be applied: supply either stats or training set to use standardization')

        self.transforms.append(FeatureStandartizatoin(stats['mean'], stats['std']))

    def _get_features(self, datapoint_name, folder_elements):
        """Get mesh vertices for given datapoint with given file list of datapoint subfolder"""
        # sequence = pattern.panel_as_sequence(self.config['panel_name'])
        pattern_nn = self._read_pattern(datapoint_name, folder_elements)
        # return random panel from a pattern
        return pattern_nn[torch.randint(pattern_nn.shape[0], (1,))]
        
    def _get_ground_truth(self, datapoint_name, folder_elements):
        """The dataset targets AutoEncoding tasks -- no need for features"""
        return None

    def _pred_to_pattern(self, prediction, dataname):
        """Save predicted value for a panel to pattern object"""
        # Not using standard util for saving as One panel out of all need update

        prediction = prediction.cpu().numpy()

        if 'standardize' in self.config:
            prediction = prediction * self.config['standardize']['std'] + self.config['standardize']['mean']

        pattern = VisPattern(str(self.root_path / dataname / 'specification.json'), view_ids=False)  # with correct pattern name

        # apply new edge info
        try: 
            pattern.panel_from_sequence(self.config['panel_name'], self._unpad(prediction, 1.5))   # we can set quite high tolerance! Normal edges are quite long
        except RuntimeError as e:
            print('GarmentPanelDataset::Warning::{}: {}'.format(dataname, e))
            pass

        return pattern


class Garment2DPatternDataset(GarmentPanelDataset):
    """Dataset definition for 2D pattern autoencoder
        * features: a 'front' panel edges represented as a sequence
        * ground_truth is not used as in Panel dataset"""
    def __init__(self, root_dir, start_config={}, gt_caching=False, feature_caching=False, transforms=[]):
        super().__init__(root_dir, start_config, gt_caching=gt_caching, feature_caching=feature_caching, transforms=transforms)
        self.config['panel_len'] = self[0]['features'].shape[1]
        self.config['element_size'] = self[0]['features'].shape[2]
    
    def _get_features(self, datapoint_name, folder_elements):
        """Get mesh vertices for given datapoint with given file list of datapoint subfolder"""
        return self._read_pattern(datapoint_name, folder_elements)
        
    def _pred_to_pattern(self, prediction, dataname):
        """Convert given predicted value to pattern object"""
        prediction = prediction.cpu().numpy()
        return self._pattern_from_tenzor(dataname, prediction, self.config, supress_error=True)


class Garment3DPatternDataset(GarmentBaseDataset):
    """Dataset definition for extracting pattern from 3D garment shape:
        * features: point samples from 3D surface
        * ground truth: tensor representation of corresponding pattern"""
    
    def __init__(self, root_dir, start_config={}, gt_caching=False, feature_caching=False, transforms=[]):
        if 'mesh_samples' not in start_config:
            start_config['mesh_samples'] = 2000  # default value if not given -- a bettern gurantee than a default value in func params
        super().__init__(root_dir, start_config, gt_caching=gt_caching, feature_caching=feature_caching, transforms=transforms)
        self.config['panel_len'] = self[0]['ground_truth'].shape[1]
        self.config['element_size'] = self[0]['ground_truth'].shape[2]
    
    def standardize(self, training=None, stats={}):
        """Use mean&std for normalization of output edge features & restoring input predictions.
            Accepts either of two inputs: 
            * training subset to calculate the data statistics -- the stats are only based on training subsection of the data
            * if stats info is given, it's used instead of calculating new statistics (usually when calling to restore dataset from existing experiment)
            'training' parameter has a priority: if it's given, the statistics are recalculated despte the value in 'stats' param
        """
        print('Garment3DPatternDataset::Using data normalization for features & ground truth')

        if training is not None:
            loader = DataLoader(training, batch_size=len(training), shuffle=False)
            for batch in loader:
                gt_mean, gt_stds = self._get_stats(batch['ground_truth'], padded=True)
                feature_mean, feature_stds = self._get_stats(batch['features'], padded=False)
                # only one batch out there anyway
                break

            self.config['standardize'] = {
                'mean' : gt_mean.cpu().numpy(), 'std': gt_stds.cpu().numpy(), 
                'f_mean' : feature_mean.cpu().numpy(), 'f_std': feature_stds.cpu().numpy()}
            stats = self.config['standardize']
        elif not stats:
            raise ValueError('Garment3DPatternDataset::Error::Standardization cannot be applied: supply either stats or training set to use standardization')

        # print(self.config['standardize'])

        self.transforms.append(GTtandartizatoin(stats['mean'], stats['std']))
        self.transforms.append(FeatureStandartizatoin(stats['f_mean'], stats['f_std']))

    def _get_features(self, datapoint_name, folder_elements):
        """Get mesh vertices for given datapoint with given file list of datapoint subfolder"""
        points = self._sample_points(datapoint_name, folder_elements)
        return points  # return in 3D
        
    def _get_ground_truth(self, datapoint_name, folder_elements):
        """Get the pattern representation"""
        return self._read_pattern(datapoint_name, folder_elements)

    def _pred_to_pattern(self, prediction, dataname):
        """Convert given predicted value to pattern object"""
        prediction = prediction.cpu().numpy()
        return self._pattern_from_tenzor(dataname, prediction, self.config, supress_error=True)


class ParametrizedShirtDataSet(BaseDataset):
    """
    For loading the data of "Learning Shared Shape Space.." paper
    """
    
    def __init__(self, root_dir, start_config={'num_verts': 'all'}, gt_caching=False, feature_caching=False, transforms=[]):
        """
        Args:
            root_dir (string): Directory with all the t-shirt examples as subfolders
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # datapoint folder structure
        self.mesh_filename = 'shirt_mesh_r.obj'
        self.pattern_params_filename = 'shirt_info.txt'
        self.features_filename = 'visfea.mat'
        self.garment_3d_filename = 'shirt_mesh_r_tmp.obj'

        super().__init__(root_dir, start_config, gt_caching=gt_caching, feature_caching=feature_caching, transforms=transforms)
        
    def save_prediction_batch(self, predictions, datanames, save_to):
        """Saves predicted params of the datapoint to the original data folder.
            Returns list of paths to files with predictions"""
        
        prediction_files = []
        for prediction, name in zip(predictions, datanames):
            path_to_prediction = Path(save_to) / name
            path_to_prediction.mkdir(parents=True, exist_ok=True)
            
            prediction = prediction.tolist()
            with open(path_to_prediction / self.pattern_params_filename, 'w+') as f:
                f.writelines(['0\n', '0\n', ' '.join(map(str, prediction))])
                print ('Saved ' + name)
            prediction_files.append(str(path_to_prediction / self.pattern_params_filename))
        return prediction_files

    # ------ Data-specific basic functions  -------
    def _clean_datapoint_list(self):
        """Remove non-datapoints subfolders, failing cases, etc. Children are to override this function when needed"""
        self.datapoints_names.remove('P3ORMPBNJJAJ')
    
    def _get_features(self, datapoint_name, folder_elements=None):
        """features parameters from a given datapoint subfolder"""
        assert (self.root_path / datapoint_name / self.garment_3d_filename).exists(), datapoint_name
        
        verts, _ = igl.read_triangle_mesh(str(self.root_path / datapoint_name / self.garment_3d_filename))
        
        if self.config['num_verts'] == 'all':
            return verts.ravel()
        
        return verts[:self.config['num_verts']].ravel()
        
    def _get_ground_truth(self, datapoint_name, folder_elements=None):
        """9 pattern size parameters from a given datapoint subfolder"""
        assert (self.root_path / datapoint_name / self.pattern_params_filename).exists(), datapoint_name
        
        # assuming that we need the numbers from the last line in file
        with open(self.root_path / datapoint_name / self.pattern_params_filename) as f:
            lines = f.readlines()
            params = np.fromstring(lines[-1],  sep = ' ')
        return params
       

if __name__ == "__main__":

    # data_location = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
    system = Properties('./system.json')
    # dataset_folder = 'data_1000_skirt_4_panels_200616-14-14-40'
    dataset_folder = 'data_1000_tee_200527-14-50-42_regen_200612-16-56-43'

    data_location = Path(system['output']) / dataset_folder

    dataset = Garment3DPatternDataset(data_location)

    print(len(dataset), dataset.config)
    print(dataset[0]['name'], dataset[0]['features'].shape, dataset[0]['ground_truth'].shape)

    print(dataset[5]['features'])

    datawrapper = DatasetWrapper(dataset)
    datawrapper.new_split(10, 10, 300)

    datawrapper.standardize_data()

    print(dataset[0]['ground_truth'])
    print(dataset[5]['features'])