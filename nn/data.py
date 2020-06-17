import numpy as np
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import openmesh as om

# ---------------------- Main Wrapper ------------------
class DatasetWrapper(object):
    """Resposible for keeping dataset, its splits, loaders & processing routines"""
    def __init__(self, in_dataset):
        self.dataset = in_dataset

        self.training = in_dataset
        self.validation = None
        self.test = None

        self.loader_train = None
        self.loader_validation = None
        self.loader_test = None

        self.split_info = {
            'random_seed': None, 
            'valid_percent': None, 
            'test_percent': None
        }
    
    def new_split(self, valid_percent, test_percent=None, random_seed=None):
        """Creates train/validation or train/validation/test splits
            depending on provided parameters
            """
        self.split_info['random_seed'] = random_seed if random_seed else int(time.time())
        self.split_info.update(valid_percent=valid_percent, test_percent=test_percent)
        
        return self.load_split()

    def load_split(self, split_info=None):
        """Get the split by provided parameters. Can be used to reproduce splits on the same dataset"""
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
        
        return self.training, self.validation, self.test

    def save_split(self, path):
        """Save split to external file"""
        pass

    def new_loaders(self, batch_size, shuffle_train=True):
        """Create loaders for current data split"""
        self.loader_train = DataLoader(self.training, batch_size, shuffle=shuffle_train)

        self.loader_validation = DataLoader(self.validation, batch_size) if self.validation else None

        self.loader_test = DataLoader(self.test, batch_size) if self.test else None

        return self.loader_train, self.loader_validation, self.loader_test

# ------------------ Transforms ----------------
# Custom transforms -- to tensor
class SampleToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        features, params = sample['features'], sample['pattern_params']
        
        return {
            'features': torch.from_numpy(features).float(), 
            'pattern_params': torch.from_numpy(params).float(), 
            'name': sample['name']
        }


# --------------------- Datasets -------------------------
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
        
        # read the pattern parameters
        pattern_parameters = self.read_pattern_params(datapoint_name)
        
        sample = {'features': vert_list.ravel(), 'pattern_params': pattern_parameters, 'name': datapoint_name}
        
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
    
    def feature_size(self):
        return 12252 * 3
        

if __name__ == "__main__":

    data_location = r'D:\Data\CLOTHING\Learning Shared Shape Space_shirt_dataset_rest'
    dataset = ParametrizedShirtDataSet(
        Path(data_location), SampleToTensor())

    print(len(dataset))
    print(dataset[100]['features'])
    print(dataset[100]['features'].shape)
    print(dataset[0]['pattern_params'].shape)
    # print (dataset[1000])

    # loader = DataLoader(dataset, 10, shuffle=True)

    # test if all elements are avaliable
    for name in dataset.datapoints_names:
        if not (dataset.root_path / name / dataset.garment_3d_filename).exists():
            print(name)