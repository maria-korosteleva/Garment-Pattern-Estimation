import json

from pathlib import Path
import os
from datetime import datetime

class PatternWrapper():
    """
    Processing of pattern template in custom JSON format
    * Loading
    * Changing the pattern parameters
    * Converting to various needed representations
    """
    def __init__(self, template_file, name="", shuffle=""):
        
        self.template_file = template_file
        with open(template_file, 'r') as f_json:
            self.pattern = json.load(f_json)

        if name:
            self.name = name
        elif shuffle:
            self.name = 'sample'
        else: 
            self.name = 'base'

    def save_pattern(self, path):
        log_dir = Path(path) / self.name
        os.makedirs(log_dir)

        # specification
        with open(log_dir / 'specification.json', 'w') as f_json:
            json.dump(self.pattern, f_json)
        
        # visualtisation TODO





if __name__ == "__main__":

    base_path = Path('D:/GK-Pattern-Data-Gen/')
    pattern = PatternWrapper(base_path / 'Patterns' / 'Flat Skirt base.json')

    print (pattern.pattern['parameters'])

    log_folder = 'data_test_' + datetime.now().strftime('%y%m%d-%H-%M')
    os.makedirs(base_path / log_folder)
    pattern.save_pattern(base_path / log_folder)
    



