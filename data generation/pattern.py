import json
import svgwrite
from svglib import svglib
from reportlab.graphics import renderPM

from pathlib import Path
import os
from datetime import datetime
import string
import random

class PatternWrapper():
    """
    Processing of pattern template in custom JSON format
    * Loading
    * Changing the pattern parameters
    * Converting to various needed representations
    """
    def __init__(self, template_file, shuffle=""):
        
        self.template_file = Path(template_file)
        self.name = self.__get_name(self.template_file.stem, shuffle)
        
        with open(template_file, 'r') as f_json:
            self.pattern = json.load(f_json)
        
    def save_pattern(self, path, to_subfolder=True):
        # log context
        if to_subfolder:
            log_dir = Path(path) / self.name
            os.makedirs(log_dir)
            secification_file = log_dir / 'specification.json'
            svg_file = log_dir / 'pattern.svg'
            png_file = log_dir / 'pattern.png'
        else:
            secification_file = Path(path) / (self.name + '_specification.json')
            svg_file = Path(path) / (self.name + '_pattern.svg')
            png_file = Path(path) / (self.name + '_pattern.png')

        # Save specification
        with open(secification_file, 'w') as f_json:
            json.dump(self.pattern, f_json)
        # visualtisation
        self.__save_as_image(svg_file, png_file)
    
    
    # --------- Main Functionality ----------

    # -------- Utils ---------
    def __save_as_image(self, svg_filename, png_filename):
        """Saves current pattern in svg and png format for visualization"""
        
        dwg = svgwrite.Drawing(svg_filename.as_posix(), profile='tiny', 
                                size = ("800px", "600px"))      # might want to re-check sizing \ rescale with the size of the pattern
        dwg.add(dwg.line((0, 0), (10, 0), stroke=svgwrite.rgb(10, 10, 16, '%')))
        dwg.add(dwg.text('hello world', insert=(0, 20), fill='red'))    # name the panels drawn
        dwg.save()

        # to png
        svg_pattern = svglib.svg2rlg(svg_filename.as_posix())
        renderPM.drawToFile(svg_pattern, png_filename, fmt='png')


    def __id_generator(self, size=10, chars=string.ascii_uppercase + string.digits):
        """Generated a random string of a given size, see 
        https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits 
        """
        return ''.join(random.choices(chars, k=size))

    def __get_name(self, prefix, shuffle):
        name = prefix
        if shuffle:
            name = name + '_' + self.__id_generator()
        return name


if __name__ == "__main__":

    base_path = Path('D:/GK-Pattern-Data-Gen/')
    pattern = PatternWrapper(base_path / 'Patterns' / 'Flat Skirt base.json', shuffle=True)
    print (pattern.pattern['parameters'])

    # log to file
    log_folder = 'data_test_' + datetime.now().strftime('%y%m%d-%H-%M')
    os.makedirs(base_path / log_folder)
    pattern.save_pattern(base_path / log_folder, to_subfolder=False)
    



