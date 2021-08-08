import copy
from collections import OrderedDict
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import sys

if sys.version_info[0] >= 3:
    from scipy.spatial.transform import Rotation as scipy_rot  # Not available in scipy 0.19.1 installed for Maya

# My modules
from pattern.core import panel_spec_template
from pattern.wrappers import VisPattern


# ------- Custom Errors --------
class EmptyPanelError(Exception):
    pass


class InvalidPatternDefError(Exception):
    """
        The given pattern definition (e.g. numeric representation) is not self-consistent.
        Examples: stitches refer to non-existing edges
    """
    def __init__(self, pattern_name='', message=''):
        self.message = 'Pattern {} is invalid'.format(pattern_name)
        if message:
            self.message += ': ' + message
        super().__init__(self.message)


# -------- Pattern Interface -----
class NNSewingPattern(VisPattern):
    """
        Interface to Sewing patterns with Neural Net friendly representation
    """
    def __init__(self, pattern_file=None, view_ids=False, panel_classifier=None, template_name=None):
        """
            `template_name` is need to use `panel_classifier` for panel ordering
        """
        self.panel_classifier = panel_classifier
        self.template_name = template_name

        super().__init__(pattern_file=pattern_file, view_ids=view_ids)        

    def pattern_as_tensors(
            self, 
            pad_panels_to_len=None, pad_panels_num=None, pad_stitches_num=None,
            with_placement=False, with_stitches=False, with_stitch_tags=False):
        """Return pattern in format suitable for NN inputs/outputs
            * 3D tensor of panel edges
            * 3D tensor of panel's 3D translations
            * 3D tensor of panel's 3D rotations
        Parameters to control padding: 
            * pad_panels_to_len -- pad the list edges of every panel to this number of edges
            * pad_panels_num -- pad the list of panels of the pattern to this number of panels
        """
        if sys.version_info[0] < 3:
            raise RuntimeError('BasicPattern::Error::pattern_as_tensors() is only supported for Python 3.6+ and Scipy 1.2+')
        
        # get panel ordering
        panel_order = self.panel_order(pad_to_len=pad_panels_num)

        # Calculate max edge count among panels -- if not provided
        panel_lens = [len(self.pattern['panels'][name]['edges']) if name is not None else 0 for name in panel_order]
        max_len = pad_panels_to_len if pad_panels_to_len is not None else max(panel_lens)

        # Main info per panel
        panel_seqs, panel_translations, panel_rotations = [], [], []
        for panel_name in panel_order:
            if panel_name is not None:
                edges, rot, transl = self.panel_as_numeric(panel_name, pad_to_len=max_len)
            else:  # empty panel
                edges, rot, transl = self._empty_panel(max_len)
            panel_seqs.append(edges)
            panel_translations.append(transl)
            panel_rotations.append(rot)

        # Stitches info. Order of stitches doesn't matter
        stitches_num = len(self.pattern['stitches']) if pad_stitches_num is None else pad_stitches_num
        if stitches_num < len(self.pattern['stitches']):
            raise ValueError(
                'BasicPattern::Error::requested number of stitches {} is less the number of stitches {} in pattern {}'.format(
                    stitches_num, len(self.pattern['stitches']), self.name
                ))
        
        # Padded value is zero allows to treat the whole thing as index array
        # But need care when using -- as indexing will not crush when padded values are not filtered
        stitches_indicies = np.zeros((2, stitches_num), dtype=np.int) 
        if with_stitch_tags:
            # padding happens automatically, if panels are padded =)
            stitch_tags = self.stitches_as_tags()
            tags_per_edge = np.zeros((len(panel_seqs), len(panel_seqs[0]), stitch_tags.shape[-1]))
        for idx, stitch in enumerate(self.pattern['stitches']):
            for id_side, side in enumerate(stitch):
                panel_id = panel_order.index(side['panel'])
                edge_id = side['edge']
                stitches_indicies[id_side][idx] = panel_id * max_len + edge_id  # pattern-level edge id
                if with_stitch_tags:
                    tags_per_edge[panel_id][edge_id] = stitch_tags[idx]

        # format result as requested
        result = [np.stack(panel_seqs), np.array(panel_lens)]
        result.append(len(self.pattern['panels']))  # actual number of panels 
        if with_placement:
            result.append(np.stack(panel_rotations))
            result.append(np.stack(panel_translations))
        if with_stitches:
            result.append(stitches_indicies)
            result.append(len(self.pattern['stitches']))  # actual number of stitches
        if with_stitch_tags:
            result.append(tags_per_edge)

        return tuple(result) if len(result) > 1 else result[0]

    def pattern_from_tensors(
            self, pattern_representation, 
            panel_rotations=None, panel_translations=None, stitches=None,
            padded=False):
        """Create panels from given panel representation. 
            Assuming that representation uses cm as units"""
        if sys.version_info[0] < 3:
            raise RuntimeError('BasicPattern::Error::pattern_from_tensors() is only supported for Python 3.6+ and Scipy 1.2+')

        # Invalidate parameter & constraints values
        self._invalidate_all_values()

        # remove existing panels -- start anew
        self.pattern['panels'] = {}
        in_panel_order = []
        new_panel_ids = [None] * len(pattern_representation)  # for correct stitches assignment in case of empty panels in-between
        for idx in range(len(pattern_representation)):
            panel_name = 'panel_' + str(idx) if self.panel_classifier is None else self.panel_classifier.class_name(idx)
            
            try:
                self.panel_from_numeric(
                    panel_name, 
                    pattern_representation[idx], 
                    rotation=panel_rotations[idx] if panel_rotations is not None else None,
                    translation=panel_translations[idx] if panel_translations is not None else None,
                    padded=padded)
                in_panel_order.append(panel_name)
                new_panel_ids[idx] = len(in_panel_order) - 1
            except EmptyPanelError as e:
                # Found an empty panel in the input -- moving on to the next one
                pass

        self.pattern['panel_order'] = in_panel_order  # save the incoming panel order

        # remove existing stitches -- start anew
        self.pattern['stitches'] = []
        if stitches is not None and len(stitches) > 0:
            if not padded:
                # TODO implement mapping of pattern-level edge ids -> (panel_id, edge_id) for panels with different number of edges
                raise NotImplementedError('BasicPattern::Recovering stitches for unpadded pattern is not supported')
            
            edges_per_panel = pattern_representation.shape[1]
            for stitch_id in range(stitches.shape[1]):
                stitch_object = []
                for side_id in range(stitches.shape[0]):
                    pattern_edge_id = stitches[side_id][stitch_id]
                    in_panel_id = int(pattern_edge_id // edges_per_panel)
                    if in_panel_id > (len(pattern_representation) - 1) or new_panel_ids[in_panel_id] is None:  # validity of stitch definition
                        raise InvalidPatternDefError(self.name, 'stitch {} referes to non-existing panel {}'.format(stitch_id, in_panel_id))
                    stitch_object.append(
                        {
                            "panel": in_panel_order[new_panel_ids[in_panel_id]],  # map to names of filteres non-empty panels
                            "edge": int(pattern_edge_id % edges_per_panel), 
                        }
                    )
                self.pattern['stitches'].append(stitch_object)
        else:
            print('BasicPattern::Warning::{}::Panels were updated but new stitches info was not provided. Stitches are removed.'.format(self.name))

    def panel_as_numeric(self, panel_name, pad_to_len=None):
        """
            Represent panel as sequence of edges with each edge as vector of fixed length plus the info on panel placement.
            * Edges are returned in additive manner: 
                each edge as a vector that needs to be added to previous edges to get a 2D coordinate of end vertex
            * Panel translation is represented with "universal" heuristic -- as translation of midpoint of the top-most bounding box edge
            * Panel rotation is returned as is but in quaternions

            NOTE: 
                The conversion uses the panels edges order as is, and 
                DOES NOT take resposibility to ensure the same traversal order of panel edges is used across datapoints of similar garment type.
                (the latter is done on sampling or on load)
        """
        if sys.version_info[0] < 3:
            raise RuntimeError('BasicPattern::Error::panel_as_numeric() is only supported for Python 3.6+ and Scipy 1.2+')

        panel = self.pattern['panels'][panel_name]
        vertices = np.array(panel['vertices'])
        
        # -- Construct the edge sequence in the recovered order --
        edge_sequence = [self._edge_as_vector(vertices, edge) for edge in panel['edges']]

        # padding if requested
        if pad_to_len is not None:
            if len(edge_sequence) > pad_to_len:
                raise ValueError('BasicPattern::{}::panel {} cannot fit into requested length: {} edges to fit into {}'.format(
                    self.name, panel_name, len(edge_sequence), pad_to_len))
            for _ in range(len(edge_sequence), pad_to_len):
                edge_sequence.append(np.zeros_like(edge_sequence[0]))
        
        # ----- 3D placement convertion  ------
        # Global Translation (more-or-less stable across designs)
        translation, _ = self._panel_universal_transtation(panel_name)

        panel_rotation = scipy_rot.from_euler('xyz', panel['rotation'], degrees=True)  # pattern rotation follows the Maya convention: intrinsic xyz Euler Angles
        rotation_representation = np.array(panel_rotation.as_quat())

        return np.stack(edge_sequence, axis=0), rotation_representation, translation

    def panel_from_numeric(self, panel_name, edge_sequence, rotation=None, translation=None, padded=False):
        """ Updates or creates panel from NN-compatible numeric representation
            * Set panel vertex (local) positions & edge dictionaries from given edge sequence
            * Set panel 3D translation and orientation if given. Accepts 6-element rotation representation -- first two colomns of rotation matrix"""
        if sys.version_info[0] < 3:
            raise RuntimeError('BasicPattern::Error::panel_from_numeric() is only supported for Python 3.6+ and Scipy 1.2+')

        if padded:
            # edge sequence might be ending with pad values or the whole panel might be a mock object
            selection = ~np.all(np.isclose(edge_sequence, 0, atol=1.5), axis=1)  # only non-zero rows
            edge_sequence = edge_sequence[selection]
            if len(edge_sequence) < 3:
                # 0, 1, 2 edges are not enough to form a panel -> assuming this is a mock panel
                raise EmptyPanelError('{}::EmptyPanelError::Supplied <{}> is empty'.format(self.__class__.__name__, panel_name))

        if panel_name not in self.pattern['panels']:
            # add new panel! =)
            self.pattern['panels'][panel_name] = copy.deepcopy(panel_spec_template)

        # ---- Convert edge representation ----
        vertices = np.array([[0, 0]])  # first vertex is always at origin
        edges = []
        for idx in range(len(edge_sequence) - 1):
            edge_info = edge_sequence[idx]
            next_vert = vertices[idx] + edge_info[:2]
            vertices = np.vstack([vertices, next_vert])
            edges.append(self._edge_dict(idx, idx + 1, edge_info[2:4]))

        # last edge is a special case
        idx = len(vertices) - 1
        edge_info = edge_sequence[-1]
        fin_vert = vertices[-1] + edge_info[:2]
        if all(np.isclose(fin_vert, 0, atol=3)):  # 3 cm per coordinate is a tolerable error
            edges.append(self._edge_dict(idx, 0, edge_info[2:4]))
        else:
            print('BasicPattern::Warning::{} with panel {}::Edge sequence do not return to origin. '
                  'Creating extra vertex'.format(self.name, panel_name))
            vertices = np.vstack([vertices, fin_vert])
            edges.append(self._edge_dict(idx, idx + 1, edge_info[2:4]))

        # update panel itself
        panel = self.pattern['panels'][panel_name]
        panel['vertices'] = vertices.tolist()
        panel['edges'] = edges

        # ----- 3D placement setup --------
        if rotation is not None:
            rotation_obj = scipy_rot.from_quat(rotation)
            panel['rotation'] = rotation_obj.as_euler('xyz', degrees=True).tolist()

        if translation is not None:
            # we are getting translation of 3D top-midpoint (aka 'universal translation')
            # convert it to the translation from the origin 
            _, transl_origin = self._panel_universal_transtation(panel_name)

            shift = np.append(transl_origin, 0)  # to 3D
            panel_rotation = scipy_rot.from_euler('xyz', panel['rotation'], degrees=True)
            comenpensating_shift = - panel_rotation.as_matrix().dot(shift)
            translation = translation + comenpensating_shift

            panel['translation'] = translation.tolist()
        
    def stitches_as_tags(self, panel_order=None, pad_to_len=None):
        """For every stitch, assign an approximate identifier (tag) of the stitch to the edges that are part of that stitch
            * tags are calculated as ~3D locations of the stitch when the garment is draed on the body in T-pose
            * It's calculated as average of the participating edges' endpoint -- Although very approximate, this should be enough
            to separate stitches from each other and from free edges
        Return
            * List of stitch tags for every stitch in the panel
            TODO Update description
            * per-edge, per-panel list of 3D tags
            * If pad_to_len is provided, per-edge lists of tags are padded to this len s.t. all panels have the same number of (padded) edges

        """
        # NOTE stitch tags values are independent from the choice of origin & edge order within a panel
        # iterate over stitches
        stitch_tags = []
        for stitch in self.pattern['stitches']:
            edge_tags = np.empty((2, 3))  # two 3D tags per edge
            for side_idx, side in enumerate(stitch):
                panel = self.pattern['panels'][side['panel']]
                edge_endpoints = panel['edges'][side['edge']]['endpoints']
                # get 2D locations of participating vertices -- per panel
                edge_endpoints = np.array([
                    panel['vertices'][edge_endpoints[side]] for side in [0, 1]
                ])
                # Get edges midpoints (2D)
                edge_mean = edge_endpoints.mean(axis=0)

                # calculate their 3D locations
                edge_tags[side_idx] = self._point_in_3D(edge_mean, panel['rotation'], panel['translation'])

            # take average
            stitch_tags.append(edge_tags.mean(axis=0))

        return np.array(stitch_tags)

    def _edge_dict(self, vstart, vend, curvature):
        """Convert given info into the proper edge dictionary representation"""
        edge_dict = {'endpoints': [vstart, vend]}
        if not all(np.isclose(curvature, 0, atol=0.01)):  # 0.01 is tolerable error for local curvature coords
            edge_dict['curvature'] = curvature.tolist()
        return edge_dict

    def _empty_panel(self, max_edge_num):
        """ Shape, rotation, and translation for empty panels"""
        # edge is 4-elem vector, 4 rotation element for quaternion, 3 element for world translation
        return np.zeros((max_edge_num, 4)), np.zeros(4), np.zeros(3)

    # ordering of panels according to classification
    def panel_order(self, force_update=False, pad_to_len=None):
        """
            Return order of panels either 
                * according to the one provided in the pattern spec
                * According to external panels classification if self.panel_classifier is set!
            Note: 'None' represent empty panels at that place of ordered elements

            Reloading 'panel_order' instead of 'define_panel_order' to preserve order from file 
                if self.panel_classifier is not defined and 'force_update' is false
        """
        if self.panel_classifier is None or self.template_name is None:
            # preserves the order is given in pattern spec!
            order = super().panel_order(force_update=force_update)
            
        else:  
            # NOTE: re-evaluate even if `force_update` flag is false
            # as we need update even if the pattern spec already contains some order

            # construct the order according to class indices
            # -None- represents empty panels-placeholders
            order = [None] * len(self.panel_classifier)
            for panel_name in self.pattern['panels']:
                class_idx = self.panel_classifier.class_idx(self.template_name, panel_name)
                order[class_idx] = panel_name
        
        # Additionally pad to requested value if given
        if pad_to_len is not None:
            if pad_to_len < len(order):
                raise ValueError(
                    f'{self.__class__.__name__}::{self.name}::Error::Requested max num of panels {pad_to_len} '
                    f'is smaller then evaluated number of panels {len(order)}')
            order += [None] * (pad_to_len - len(order))

        # Remember the order for future reference
        self.pattern['panel_order'] = order

        return order


# -------- Panel classification Interface -----
class PanelClasses():
    """ Interface to access panel classification by role """
    def __init__(self, classes_file):

        self.filename = classes_file

        with open(classes_file, 'r') as f:
            # preserve the order of classes names
            self.classes = json.load(f, object_pairs_hook=OrderedDict)

        self.names = list(self.classes.keys())
        
        self.panel_to_idx = {}
        for idx, class_name in enumerate(self.classes):
            panels_list = self.classes[class_name]
            for panel in panels_list:
                self.panel_to_idx[tuple(panel)] = idx
        
    def __len__(self):
        return len(self.classes)

    def class_idx(self, template, panel):
        """
            Return idx of class for given panel (name) from given template(name)
        """
        # TODO process cases when given pair does not exist in the classes
        return self.panel_to_idx[(template, panel)]

    def class_name(self, idx):
        return self.names[idx]


# ---------- test -------------
if __name__ == "__main__":
    import customconfig
    from pattern.wrappers import VisPattern

    # np.set_printoptions(precision=4, suppress=True)

    system_config = customconfig.Properties('./system.json')
    base_path = system_config['output']
    pattern = NNSewingPattern(
        Path(system_config['templates_path']) / 'basic tee' / 'tee.json', 
        panel_classifier=PanelClasses('./nn/panel_classes.json'), 
        template_name='tee')

    empty_pattern = NNSewingPattern(panel_classifier=PanelClasses('./nn/panel_classes.json'))
    print(pattern.panel_order())

    # print(pattern.stitches_as_tags())

    # print(len(pattern.pattern_as_tensors(with_placement=True, with_stitches=True, with_stitch_tags=True)))

    tensor, edge_lens, num_panels, rot, transl, stitches, stitch_num, stitch_tags = pattern.pattern_as_tensors(
        with_placement=True, with_stitches=True, with_stitch_tags=True)

    empty_pattern.pattern_from_tensors(tensor, rot, transl, stitches, padded=True)
    # print(pattern.pattern['stitches'])
    # print(empty_pattern.panel_order())

    # Save
    empty_pattern.name = pattern.name + 'from_empty_with_class' + '_' + datetime.now().strftime('%y%m%d-%H-%M-%S')
    pattern.name = pattern.name + '_with_class' + '_' + datetime.now().strftime('%y%m%d-%H-%M-%S')
    
    empty_pattern.serialize(system_config['output'], to_subfolder=True)
    pattern.serialize(system_config['output'], to_subfolder=True)
