# Libs
from pathlib import Path
from datetime import datetime
import time
import json
import os
import numpy as np

# My module
import pattern


class DatasetProperties():
    def __init__(self):
        self.properties = {}
    
    def __getitem__(self, key):
        return self.properties[key]

    def __setitem__(self, key, value):
        self.properties[key] = value
    
    def serialize(self, filename):
        with open(filename, 'w') as f_json:
            json.dump(self.properties, f_json, indent=2)


def generate(path, template_file_path, size, data_to_subfolders=True, name=""):
    path = Path(path)
    template_file_path = Path(template_file_path)

    # init properties 
    props = DatasetProperties()
    props['size'] = size
    props['template'] = template_file_path.name
    props['to_subfolders'] = data_to_subfolders

    # create log folders
    data_folder = name + '_' + template_file_path.stem + '_' \
        + datetime.now().strftime('%y%m%d-%H-%M')
    props['data_folder'] = data_folder
    path_with_dataset = path / data_folder
    os.makedirs(path_with_dataset)

    # init random seed
    timestamp = int(time.time())
    np.random.seed(timestamp)
    props['random_seed'] = timestamp

    # generate data
    start_time = time.time()
    for _ in range(size):
        new_pattern = pattern.PatternWrapper(template_file_path, 
                                             randomize=True)
        new_pattern.serialize(path_with_dataset, 
                              to_subfolder=data_to_subfolders)
    elapsed = time.time() - start_time
    props['generation_time'] = f'{elapsed:.3f} s'

    # log properties
    props.serialize(path_with_dataset / 'dataset_properties.json')
    # TODO copy template? 


if __name__ == "__main__":

    base_path = Path('D:/GK-Pattern-Data-Gen/')
    generate(base_path, base_path / 'Patterns' / 'skirt_per_panel.json', 
             size=5, 
             data_to_subfolders=True, 
             name='test')
