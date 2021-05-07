"""
    Visualize requested point cloud with attention weights per point 
    * To be run within Maya environment
"""

import os
import numpy as np
# Maya
import maya.cmds as cmds

# setup
pred_path = 'Tee-JS-segment-shuffle-orderless-125/test/jumpsuit_sleeveless'
name = 'jumpsuit_sleeveless_U0K41NJ0NJ'
# att_weight_id = 3

# code
base_path = 'C:/Users/Asus/Desktop/Garments_outs'  # local path to all logs
pred_path = os.path.join(base_path, pred_path)

# load weights
point_cloud_filepath = os.path.join(pred_path, name, (name + '_point_cloud.txt'))
weights_filepath = os.path.join(pred_path, name, (name + '_att_weights.txt'))

print(point_cloud_filepath)
print(weights_filepath)

point_cloud = np.loadtxt(point_cloud_filepath)
att_weights = np.loadtxt(weights_filepath)

# colors
# Coolors.com
color_hex = ['608059', '6D2848', 'F31400', 'FA7D00', '9975C1', '85B79D', '6F686D', 'FF715B', 'FFDF64', 'C6D4FF']
# to rgb codes
colors = np.empty((len(color_hex), 3))
for idx in range(len(color_hex)):
    colors[idx] = np.array([int(color_hex[idx][i:i + 2], 16) for i in (0, 2, 4)]) / 255.0

# Leave usable colors according to the number of weights
colors = colors[:att_weights.shape[-1]]  

# create and color point bubbles
objects = []
for idx in range(len(point_cloud)):
    # create point
    sphere = cmds.sphere(r=2)
    objects.append(sphere[0])

    coords = point_cloud[idx]
    cmds.move(coords[0], coords[1], coords[2], sphere, absolute=True)

    # mix colors according to one of the weights

    if 'att_weight_id' in locals() or 'att_weight_id' in globals():
        point_weights = att_weights[idx][att_weight_id]
        point_color = colors[att_weight_id] * point_weights
    else:
        point_weights = np.expand_dims(att_weights[idx], axis=1)
        point_color = (colors * point_weights).sum(axis=0)

    print(point_color)
    
    # set color
    sh_name = name + '_' + str(idx) 
    material = cmds.shadingNode('lambert', name=sh_name, asShader=True)
    sg = cmds.sets(name="%sSG" % sh_name, empty=True, renderable=True, noSurfaceShader=True)
    cmds.connectAttr("%s.outColor" % material, "%s.surfaceShader" % sg)
    cmds.setAttr(material + ".color", point_color[0], point_color[1], point_color[2], type="double3")
    cmds.sets(sphere, forceElement=sg)

cmds.group(objects, name=name)
