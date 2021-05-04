"""
    Visualize requested point cloud with attention weights per point 
    * To be run within Maya environment
"""

import os
import numpy as np
# Maya
import maya.cmds as cmds

# setup
pred_path = 'nn_pred_210504-17-41-12/test/tee'
name = 'tee_0FX6C0VKR3'
panel_id = 0

# code
base_path = 'C:/Users/Asus/Desktop/Garments_outs'  # local path to all logs
pred_path = os.path.join(base_path, pred_path)

point_cloud_filepath = os.path.join(pred_path, name, (name + '_point_cloud.txt'))
weights_filepath = os.path.join(pred_path, name, (name + '_att_weights.txt'))

print(point_cloud_filepath)
print(weights_filepath)

point_cloud = np.loadtxt(point_cloud_filepath)
att_weights = np.loadtxt(weights_filepath)

objects = []
for idx in range(len(point_cloud)):
    # create point
    sphere = cmds.sphere(r=2)
    objects.append(sphere[0])

    coords = point_cloud[idx]
    cmds.move(coords[0], coords[1], coords[2], sphere, absolute=True)

    # color according to one of the weights
    color = np.array([1., 1., 1.]) * att_weights[idx][panel_id]
    sh_name = name + '_' + str(idx)
    
    material = cmds.shadingNode('lambert', name=sh_name, asShader=True)
    sg = cmds.sets(name="%sSG" % sh_name, empty=True, renderable=True, noSurfaceShader=True)
    cmds.connectAttr("%s.outColor" % material, "%s.surfaceShader" % sg)

    cmds.setAttr(material + ".color", color[0], color[1], color[2], type="double3")

    cmds.sets(sphere, forceElement=sg)


cmds.group(objects, name='{}_{}'.format(name, panel_id))
