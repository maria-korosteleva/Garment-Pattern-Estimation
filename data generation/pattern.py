import json
import svgwrite
from svglib import svglib
from reportlab.graphics import renderPM

from pathlib import Path
import os
from datetime import datetime
import string
import random

import numpy as np

class PatternWrapper():
    """
    Processing of pattern template in custom JSON format
    * Loading
    * Changing the pattern parameters
    * Converting to various needed representations
    """

    # ------------ Interface -------------
    def __init__(self, template_file, randomize=""):
        
        self.template_file = Path(template_file)
        self.name = self.__get_name(self.template_file.stem, randomize)
        
        with open(template_file, 'r') as f_json:
            self.template = json.load(f_json)
        self.pattern = self.template['pattern']
        self.parameters = self.template['parameters']

        if randomize:
            self.__randomize_parameters()
            self.___update_pattern()
        
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
            json.dump(self.template, f_json, indent=2) 
        # visualtisation
        self.__save_as_image(svg_file, png_file)
    
    
    # --------- Pattern operations ----------
    def __randomize_parameters(self):
        # TODO account for multiple parameter types
        for parameter in self.parameters:
            param_range = self.parameters[parameter]['range']
            new_value = np.random.rand() * (param_range[1] - param_range[0]) + param_range[0]
            self.parameters[parameter]['value'] = new_value
 
    def __extend_edge(self, panel, edge, scaling_factor):
        """
        Shrinks/elongates a given edge of a given panel.
        Both vertices are updated equally ignoring the edge direction
        """
        v_id_start, v_id_end = tuple(panel['edges'][edge]['endpoints'])
        v_start, v_end = np.array(panel['vertices'][v_id_start]), \
                            np.array(panel['vertices'][v_id_end])
        v_middle = (v_start + v_end) / 2
        new_half_edge_vector = scaling_factor * (v_end - v_middle)
        v_start, v_end = v_middle - new_half_edge_vector, \
                            v_middle + new_half_edge_vector

        panel['vertices'][v_id_end] = v_end.tolist()
        panel['vertices'][v_id_start] = v_start.tolist()

        # the curvature is adjusted automatically when given in relativecoordinates

    def __normalize_panel_translation(self, panel):
        """ 
        Shifts all panel vertices s.t. panel bounding box starts at zero
        for uniformity across panels & positive coordinates
        """
        vertices = np.asarray(panel['vertices'])
        offset = np.min(vertices, axis=0)
        vertices = vertices - offset

        panel['vertices'] = vertices.tolist()

    def ___update_pattern(self):
        """
        Recalculates vertex positions and edge curves according to current parameter values
        (!) Assumes that the current pattern is a template: 
                was created with all the parameters equal to 1
        """
        # TODO add other parameter types
        # Edge length adjustments  
        for parameter in self.template['parameter_order']:
            value = self.parameters[parameter]['value']
            for panel_influence in self.parameters[parameter]['influence']:
                print (parameter, panel_influence['panel'], panel_influence['edge_list'])

                panel = self.pattern['panels'][panel_influence['panel']]
                for edge in panel_influence['edge_list']:
                    self.__extend_edge(panel, edge, value)

                self.__normalize_panel_translation(panel)

    def ___calc_control_coord(self, start, end, control_scale):
        """
        Derives absolute coordinates of Bezier control point given as an offset 
        """
        control_start = control_scale[0] * (start + end)

        edge = end - start
        edge_perp = np.array([-edge[1], edge[0]])
        control_point = control_start + control_scale[1] * edge_perp

        return control_point

    # -------- Drawing ---------

    def ___draw_a_panel(self, drawing, panel_name, offset=[0, 0], scaling=1):
        """
        Adds a requested panel to the svg drawing with given offset and scaling
        Returns the lower-right vertex coordinate for the convenice of future offsetting
        """

        panel = self.pattern['panels'][panel_name]
        vertices = np.asarray(panel['vertices'], dtype=int) * scaling + offset

        start = vertices[panel['edges'][0]['endpoints'][0]]
        path = drawing.path(['M', start[0], start[1]], stroke='black', fill='rgb(255,217,194)')  
        for edge in panel['edges']:
            # assumes that edges are correctly oriented to form a closed loop when summed
            start = vertices[edge['endpoints'][0]]
            end = vertices[edge['endpoints'][1]]
            if ('curvature' in edge):
                control_scale = edge['curvature']
                control_point = self.___calc_control_coord(start, end, control_scale)
                path.push(['Q', control_point[0], control_point[1], end[0], end[1]]) 
            else:
                path.push(['L', end[0], end[1]])

            # TODO add darts visualization here!
            
        path.push('z')
        drawing.add(path)

        # name the panel
        panel_center = np.mean(vertices, axis=0)
        drawing.add(drawing.text(panel_name, insert=panel_center, fill='blue'))   

        return np.max(vertices[:, 0]), np.max(vertices[:, 1])

    def __save_as_image(self, svg_filename, png_filename):
        """Saves current pattern in svg and png format for visualization"""
        
        dwg = svgwrite.Drawing(svg_filename.as_posix(), profile='tiny')
        base_offset = [40, 40]
        panel_offset = [0, 0]
        for panel in self.pattern['panels']:
            panel_offset = self.___draw_a_panel(dwg, panel, 
                                            offset=[panel_offset[0] + base_offset[0], base_offset[1]],
                                            scaling=10) 

        # final sizing & save
        dwg['width'] = str(panel_offset[0] + base_offset[0]) + 'px'
        dwg['height'] = str(panel_offset[1] + base_offset[1]) + 'px'
        dwg.save(pretty=True)

        # to png
        svg_pattern = svglib.svg2rlg(svg_filename.as_posix())
        renderPM.drawToFile(svg_pattern, png_filename, fmt='png')

    # -------- Utils ---------

    def __id_generator(self, size=10, chars=string.ascii_uppercase + string.digits):
        """Generated a random string of a given size, see 
        https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits 
        """
        return ''.join(random.choices(chars, k=size))

    def __get_name(self, prefix, randomize):
        name = prefix
        if randomize:
            name = name + '_' + self.__id_generator()
        return name


if __name__ == "__main__":

    base_path = Path('D:/GK-Pattern-Data-Gen/')
    pattern = PatternWrapper(base_path / 'Patterns' / 'skirt_per_panel.json', randomize=True)
    # print (pattern.pattern['panels'])

    # log to file
    log_folder = 'data_random_' + datetime.now().strftime('%y%m%d-%H-%M')
    os.makedirs(base_path / log_folder)
    pattern.save_pattern(base_path / log_folder, to_subfolder=False)
    



