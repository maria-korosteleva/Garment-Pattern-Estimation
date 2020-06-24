import numpy as np
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import openmesh as om

# My modules
from customconfig import Properties
from pattern.core import ParametrizedPattern

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

    def save_split(self, path):
        """Save split to external file"""
        pass

    # --------- Managing predictions on this data ---------
    def predict(self, model, section='test', single_batch=False):
        """Save model predictions on the given dataset section"""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        with torch.no_grad():
            loader = self.get_loader(section)
            if loader:
                if single_batch:
                    batch = next(iter(loader))    # might have some issues, see https://github.com/pytorch/pytorch/issues/1917
                    features = batch['features'].to(device)
                    self.dataset.save_prediction_batch(model(features), batch['name'])
                else:
                    for batch in loader:
                        features = batch['features'].to(device)
                        self.dataset.save_prediction_batch(model(features), batch['name'])

# ------------------ Transforms ----------------
# Custom transforms -- to tensor
class SampleToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        features, params = sample['features'], sample['ground_truth']
        
        return {
            'features': torch.from_numpy(features).float(), 
            'ground_truth': torch.from_numpy(params).float(), 
            'name': sample['name']
        }


# --------------------- Datasets -------------------------

class GarmentParamsDataset(Dataset):
    """
    For loading the custom generated data & predicting generated parameters
    """
    
    def __init__(self, root_dir, *argtranforms):
        """
        Args:
            root_dir (string): Directory with all examples as subfolders
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_path = Path(root_dir)
        self.name = self.root_path.name
        self.dataset_props = Properties(self.root_path / 'dataset_properties.json')
        if not self.dataset_props['to_subfolders']:
            raise NotImplementedError('Working with datasets with all satapopints ')

        # list of items = subfolders
        root, dirs, files = next(os.walk(self.root_path))
        to_ignore = ['renders']
        self.datapoints_names = [directory for directory in dirs if directory not in to_ignore]
        self._ingnore_fails()

        # Use default tensor transform + the ones from input
        self.transform = transforms.Compose([SampleToTensor()] + list(argtranforms))

        elem = self[0]
        self.feature_size = elem['features'].shape[0]
        self.ground_truth_size = elem['ground_truth'].shape[0]

    def update_transform(self, transform):
        """apply new transform when loading the data"""
        self.transform = transform
               
    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints_names)   

    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data. 
        Does not support list indexing"""
        
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()
            
        datapoint_name = self.datapoints_names[idx]
        folder_elements = [file.name for file in (self.root_path / datapoint_name).glob('*')]  # all files in this directory

        vert_list = self._read_verts(datapoint_name, folder_elements)

        # DEBUG
        vert_list = vert_list[:500]
        
        # read the pattern parameters
        pattern_parameters = self._read_pattern_params(datapoint_name, folder_elements)
        
        sample = {'features': vert_list.ravel(), 'ground_truth': pattern_parameters, 'name': datapoint_name}
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample

    def save_prediction_batch(self, predicted_params, names):
        """Saves predicted params of the datapoint to the original data folder"""
        raise NotImplementedError()

        # for prediction, name in zip(predicted_params, names):
        #     path_to_prediction = self.root_path / '..' / 'predictions' / name
        #     try:
        #         os.makedirs(path_to_prediction)
        #     except OSError:
        #         pass
            
        #     prediction = prediction.tolist()
        #     with open(path_to_prediction / self.pattern_params_filename, 'w+') as f:
        #         f.writelines(['0\n', '0\n', ' '.join(map(str, prediction))])
        #         print ('Saved ' + name)

    # ------ Private --------
    def _ingnore_fails(self):
        """Remove all elements marked as failure from the datapoint list"""
        fails_dict = self.dataset_props['sim']['stats']['fails']
        # TODO allow not to ignore some of the subsections
        for subsection in fails_dict:
            for fail in fails_dict[subsection]:
                try:
                    self.datapoints_names.remove(fail)
                    print('Dataset:: {} ignored'.format(fail))
                except ValueError:  # if fail was already removed based on previous failure subsection
                    pass

    def _read_verts(self, datapoint_name, folder_elements):
        """Get mesh vertices for given datapoint with given file list of datapoint subfolder"""
        obj_list = [file for file in folder_elements if 'sim.obj' in file]
        if not obj_list:
            raise RuntimeError('Dataset:Error: geometry file *sim.obj not found for {}'.format(datapoint_name))
        
        mesh = om.read_trimesh(str(self.root_path / datapoint_name / obj_list[0]))
        
        return mesh.points()
        
    def _read_pattern_params(self, datapoint_name, folder_elements):
        """9 pattern size parameters from a given datapoint subfolder"""
        spec_list = [file for file in folder_elements if 'specification.json' in file]
        if not spec_list:
            raise RuntimeError('Dataset:Error: *specification.json not found for {}'.format(datapoint_name))
        
        pattern = ParametrizedPattern(self.root_path / datapoint_name / spec_list[0])

        return np.array(pattern.param_values_list())
   

class ParametrizedShirtDataSet(Dataset):
    """
    For loading the data of "Learning Shared Shape Space.." paper
    """
    
    def __init__(self, root_dir, *argtranforms):
        """
        Args:
            root_dir (string): Directory with all the t-shirt examples as subfolders
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_path = Path(root_dir)
        self.name = self.root_path.name
        
        # list of items = subfolders
        self.datapoints_names = next(os.walk(self.root_path))[1]
        
        # remove non-valid element
        self.datapoints_names.remove('P3ORMPBNJJAJ')
        
        # datapoint folder structure
        self.mesh_filename = 'shirt_mesh_r.obj'
        self.pattern_params_filename = 'shirt_info.txt'
        self.features_filename = 'visfea.mat'
        self.garment_3d_filename = 'shirt_mesh_r_tmp.obj'

        # Use default tensor transform + the ones from input
        self.transform = transforms.Compose([SampleToTensor()] + list(argtranforms))

    def update_transform(self, transform):
        """apply new transform when loading the data"""
        self.transform = transform
               
    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints_names)   
    
    def read_verts(self, datapoint_name):
        """features parameters from a given datapoint subfolder"""
        assert (self.root_path / datapoint_name / self.garment_3d_filename).exists(), datapoint_name
        
        mesh = om.read_trimesh(str(self.root_path / datapoint_name / self.garment_3d_filename))
        
        return mesh.points()
        
    def read_pattern_params(self, datapoint_name):
        """9 pattern size parameters from a given datapoint subfolder"""
        assert (self.root_path / datapoint_name / self.pattern_params_filename).exists(), datapoint_name
        
        # assuming that we need the numbers from the last line in file
        with open(self.root_path / datapoint_name / self.pattern_params_filename) as f:
            lines = f.readlines()
            params = np.fromstring(lines[-1],  sep = ' ')
        return params
       
    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data. 
        Does not support list indexing"""
        
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()
            
        datapoint_name = self.datapoints_names[idx]
        
        vert_list = self.read_verts(datapoint_name)

        # DEBUG
        vert_list = vert_list[:500]
        
        # read the pattern parameters
        pattern_parameters = self.read_pattern_params(datapoint_name)
        
        sample = {'features': vert_list.ravel(), 'ground_truth': pattern_parameters, 'name': datapoint_name}
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
        
    def save_prediction_batch(self, predicted_params, names):
        """Saves predicted params of the datapoint to the original data folder"""
        
        for prediction, name in zip(predicted_params, names):
            path_to_prediction = self.root_path / '..' / 'predictions' / name
            try:
                os.makedirs(path_to_prediction)
            except OSError:
                pass
            
            prediction = prediction.tolist()
            with open(path_to_prediction / self.pattern_params_filename, 'w+') as f:
                f.writelines(['0\n', '0\n', ' '.join(map(str, prediction))])
                print ('Saved ' + name)
        

if __name__ == "__main__":

    # data_location = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
    system = Properties('./system.json')
    dataset_folder = 'data_1000_skirt_4_panels_200616-14-14-40'

    data_location = Path(system['output']) / dataset_folder

    dataset = GarmentParamsDataset(data_location)

    print(len(dataset))
    print(dataset[100]['name'], dataset[100]['features'].shape, dataset[100]['ground_truth'])
    print(dataset[0]['ground_truth'].shape)
    # print (dataset[1000])

    # loader = DataLoader(dataset, 10, shuffle=True)