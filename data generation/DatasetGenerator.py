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
    def __init__(self, template_names, size, 
                 data_to_subfolders=True, name="", random_seed=None):
        self.properties = {}
        # Init mandatory dataset properties
        self['size'] = size
        self['templates'] = template_names
        self['to_subfolders'] = data_to_subfolders
        self['name'] = name
        if random_seed is None:
            self['random_seed'] = int(time.time())  # new random seed
        else:
            self['random_seed'] = random_seed
    
    @classmethod
    def fromfile(cls, prop_file):
        with open(prop_file, 'r') as f_json:
            props = json.load(f_json) 
        if 'data_folder' in props:
            # props of the previous dataset
            props['name'] = props['data_folder'] + '_regen'

        return cls(props['templates'], 
                   props['size'],
                   props['to_subfolders'], 
                   props['name'],
                   props['random_seed'])

    def __getitem__(self, key):
        return self.properties[key]

    def __setitem__(self, key, value):
        self.properties[key] = value
    
    def serialize(self, filename):
        with open(filename, 'w') as f_json:
            json.dump(self.properties, f_json, indent=2)


def generate(path, templates_path, props):
    path = Path(path)
    # TODO modify to support multiple templates
    template_file_path = Path(templates_path) / props['templates']

    # create data folder
    data_folder = props['name'] + '_' + template_file_path.stem + '_' \
        + datetime.now().strftime('%y%m%d-%H-%M')
    props['data_folder'] = data_folder
    path_with_dataset = path / data_folder
    os.makedirs(path_with_dataset)

    # init random seed
    np.random.seed(props['random_seed'])

    # generate data
    start_time = time.time()
    for _ in range(props['size']):
        new_pattern = pattern.PatternWrapper(template_file_path, 
                                             randomize=True)
        new_pattern.serialize(path_with_dataset, 
                              to_subfolder=props['to_subfolders'])
    elapsed = time.time() - start_time
    props['generation_time'] = f'{elapsed:.3f} s'

    # log properties
    props.serialize(path_with_dataset / 'dataset_properties.json')
    # TODO copy template? 


if __name__ == "__main__":
    props = DatasetProperties(
        'skirt_per_panel.json', 
        size=5,
        data_to_subfolders=False, 
        name='test')

    props = DatasetProperties.fromfile(
        'D:/GK-Pattern-Data-Gen/test_skirt_per_panel_200324-17-09/dataset_properties.json')

    base_path = Path('D:/GK-Pattern-Data-Gen/')
    pattern_path = base_path / 'Patterns'
    generate(base_path, pattern_path, props)
