"""
    Maya script for removing faces from 3D garement model that are not visible from the outside cameras
    The goal is to imitate scanning artifacts that result in missing geometry
    Python 2.7 compatible
    * Maya 2018+
"""

from __future__ import print_function

def remove_invisible(target, camera_surface, obstacles=[]):
    """Update target 3D mesh: remove faces that are not visible from camera_surface
        * due to self-occlusion or occlusion by an obstacle

        In my context, target is usually a garment mesh, and obstacle is a body surface
        """
    # Follows the idea of self_intersect_3D() checks used in simulation pipeline

    # get garment as OpenMaya object -- though DAG

    # for every vertex of garment
        # Send rays in all directions from the currect vertex
        # TODO Random? Uniform? Same for all vertices? 
            # for every ray, check if the FIRST intersection is the camera surface -> visible
            # (Other options -- first intersection is body or self or No intersection at all)
            # if invisible, remove the vertex
            # TODO is it possible?

    print('Performing scanning imitation on {} from {} with obstacles {}'.format(target, camera_surface, obstacles))


if __name__ == "__main__":
    # Sample script that can be run within Maya for testing purposes
    # Copy the following block to Maya script editor and modify to 
    import maya.cmds as cmds
    import mayaqltools as mymaya
    reload(mymaya)

    body = cmds.ls('*f_smpl*:Mesh')[0]
    garment = cmds.ls('*tee*:Mesh')[0]
    cam_surface = cmds.ls('*camera_surface*')[0]

    mymaya.scan_imitation.remove_invisible(garment, cam_surface, [body])