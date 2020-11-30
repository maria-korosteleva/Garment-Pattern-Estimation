"""
    Maya script for removing faces from 3D garement model that are not visible from the outside cameras
    The goal is to imitate scanning artifacts that result in missing geometry
    Python 2.7 compatible
    * Maya 2018+
"""

from __future__ import print_function
from maya import OpenMaya
from maya import cmds
import numpy as np
from datetime import datetime


# TODO Move these two utils to shared place?
def get_mesh(object_name):
    """Return MFnMesh object by the object name"""
    # get object as OpenMaya object -- though DAG
    selectionList = OpenMaya.MSelectionList()
    selectionList.add(object_name)
    dag = OpenMaya.MDagPath()
    selectionList.getDagPath(0, dag)
    # as mesh
    mesh = OpenMaya.MFnMesh(dag)  # reference https://help.autodesk.com/view/MAYAUL/2017/ENU/?guid=__py_ref_class_open_maya_1_1_m_fn_mesh_html

    return mesh, dag

def test_intersect(mesh, raySource, rayVector, accelerator=None, hit_tol=None):
    """Check if given ray intersect given mesh
        * hit_tol ignores intersections that are within hit_tol from the ray source (as % of ray length) -- usefull when checking self-intersect"""
    # follow structure https://stackoverflow.com/questions/58390664/how-to-fix-typeerror-in-method-mfnmesh-anyintersection-argument-4-of-type
    maxParam = 1  # only search for intersections within given vector
    testBothDirections = False  # only in the given direction
    sortHits = False  # no need to waste time on sorting

    hitPoints = OpenMaya.MFloatPointArray()
    hitRayParams = OpenMaya.MFloatArray()
    hitFaces = OpenMaya.MIntArray()
    hit = mesh.allIntersections(
        raySource, rayVector, None, None, False, OpenMaya.MSpace.kWorld, maxParam, testBothDirections, accelerator, sortHits,
        hitPoints, hitRayParams, hitFaces, None, None, None, 1e-6)   # TODO anyIntersection
    
    if hit and hit_tol is not None:
        return any([dist > hit_tol for dist in hitRayParams])

    return hit

def sample_on_sphere(rad):
    """Uniformly sample a point on a sphere with radious rad. Return as Maya-compatible floating-point vector"""
    # Using method of (Muller 1959, Marsaglia 1972)
    # see the last one here https://mathworld.wolfram.com/SpherePointPicking.html

    uni_array = np.random.normal(size=3)

    uni_array = uni_array / np.linalg.norm(uni_array) * rad

    return OpenMaya.MFloatVector(uni_array[0], uni_array[1], uni_array[2])

def camera_surface(target, obstacles=[], vertical_scaling_factor=1.5, ground_scaling_factor=1.2):
    """Generate a (3D scanning) camera surface around provided scene"""

    # basically, draw a bounding box around the target
    bbox = np.array(cmds.exactWorldBoundingBox(obstacles + [target]))  # [xmin, ymin, zmin, xmax, ymax, zmax]
    
    top = bbox[3:]
    bottom = bbox[:3]
    center = (top + bottom) / 2
    dims = top - bottom
    dims = [max(dims[0], dims[2]) * ground_scaling_factor, dims[1] * vertical_scaling_factor]

    cube = cmds.polyCube(height=dims[1], depth=dims[0], width=dims[0], name='camera_surface')
    
    # align with center
    cmds.move(center[0], center[1], center[2], cube, absolute=True)

    # remove top and bottom faces -- as if no cameras there
    cmds.polyDelFacet( cube[0] + '.f[1]', cube[0] + '.f[3]')  # we know exact structure of default polyCube in Maya2018 & Maya2020

    return cube[0], np.max(dims)

def remove_invisible(target, obstacles=[], num_rays=30):
    """Update target 3D mesh: remove faces that are not visible from camera_surface
        * due to self-occlusion or occlusion by an obstacle

        In my context, target is usually a garment mesh, and obstacle is a body surface
        """
    # Follows the idea of self_intersect_3D() checks used in simulation pipeline
    print('Performing scanning imitation on {} with obstacles {}'.format(target, obstacles))
    
    # generate apropriate camera surface
    camera_surface_obj, ray_dist = camera_surface(target, obstacles)

    print(ray_dist)

    start_time = datetime.now()

    # get mesh objects as OpenMaya object
    target_mesh, target_dag = get_mesh(target)
    camera_surface_mesh, _ = get_mesh(camera_surface_obj)
    obstacles_meshes = [get_mesh(name)[0] for name in obstacles]

    # search for intersections
    target_accelerator = target_mesh.autoUniformGridParams()
    cam_surface_accelerator = camera_surface_mesh.autoUniformGridParams()
    obstacles_accs = [mesh.autoUniformGridParams() for mesh in obstacles_meshes]
    to_delete = []

    target_face_iterator = OpenMaya.MItMeshPolygon(target_dag)
    while not target_face_iterator.isDone():  # https://stackoverflow.com/questions/40422082/how-to-find-face-neighbours-in-maya
        # midpoint of the current face -- start of all the rays
        face_mean = OpenMaya.MFloatPoint(target_face_iterator.center(OpenMaya.MSpace.kWorld))
        face_id = target_face_iterator.index()

        # print('Face: {}: {}, {}, {}'.format(face_id, face_mean.x, face_mean.y, face_mean.z))

        visible = False
        # Send rays in all directions from the currect vertex
        for _ in range(num_rays):
            rayDir = sample_on_sphere(ray_dist)

            # print('Ray: {}, {}, {}'.format(rayDir.x, rayDir.y, rayDir.z))

            # Case when face is visible from camera surface
            if (test_intersect(camera_surface_mesh, face_mean, rayDir, cam_surface_accelerator)  # intesection with camera surface
                    and not any([test_intersect(mesh, face_mean, rayDir, acc,) for mesh, acc in zip(obstacles_meshes, obstacles_accs)])  # intesects any of the obstacles
                    and not test_intersect(target_mesh, face_mean, rayDir, target_accelerator, hit_tol=1e-5)):  # intersects itself
                visible = True
                break
        
        if not visible:
            to_delete.append(face_id)

        target_face_iterator.next()  # iterate!

    # Remove invisible vertices
    print('To delete {} : {}'.format(len(to_delete), to_delete))

    # Removing the faces
    delete_strs = [target + '.f[{}]'.format(face_id) for face_id in to_delete]
    cmds.polyDelFacet(tuple(delete_strs))

    passed = datetime.now() - start_time
    print('Removal finished after {}'.format(passed))
    
    cmds.delete(camera_surface_obj)  # clean-up

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