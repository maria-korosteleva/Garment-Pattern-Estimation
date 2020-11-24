"""
    Maya script for removing faces from 3D garement model that are not visible from the outside cameras
    The goal is to imitate scanning artifacts that result in missing geometry
    Python 2.7 compatible
    * Maya 2018+
"""

from __future__ import print_function
from maya import OpenMaya

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

def test_intersect(mesh, raySource, rayVector, accelerator, at_least_2=False):
    """Check if given ray intersect given mesh
        * at least 2 checks wheter there is at least two intersecting points -- usefull when checking self-intersect"""
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
    
    if hit and at_least_2:
        return hitPoints.length() > 1

    return hit

def remove_invisible(target, camera_surface, obstacles=[]):
    """Update target 3D mesh: remove faces that are not visible from camera_surface
        * due to self-occlusion or occlusion by an obstacle

        In my context, target is usually a garment mesh, and obstacle is a body surface
        """
    # Follows the idea of self_intersect_3D() checks used in simulation pipeline
    print('Performing scanning imitation on {} from {} with obstacles {}'.format(target, camera_surface, obstacles))
    
    # get target as OpenMaya object -- though DAG
    target_mesh, target_dag = get_mesh(target)
    target_vertices = OpenMaya.MPointArray()
    target_mesh.getPoints(target_vertices, OpenMaya.MSpace.kWorld)

    # other meshes
    camera_surface_mesh, _ = get_mesh(camera_surface)
    obstacles_meshes = [get_mesh(name)[0] for name in obstacles]

    # search for intersections
    target_accelerator = OpenMaya.MMeshIsectAccelParams()  
    cam_surface_accelerator = OpenMaya.MMeshIsectAccelParams() 
    obstacles_accs = [OpenMaya.MMeshIsectAccelParams() for _ in range(len(obstacles))]
    invisible_counter = 0
    self_intersect = 0
    object_intersect = 0
    to_delete_list = []

    target_face_iterator = OpenMaya.MItMeshPolygon(target_dag)
    to_delete = OpenMaya.MDGModifier()  # will collect all delete requests and execute at once
    last_deleted = -1
    while not target_face_iterator.isDone():  # https://stackoverflow.com/questions/40422082/how-to-find-face-neighbours-in-maya
        # midpoint of the current face -- start of all the rays
        face_mean = OpenMaya.MFloatPoint(target_face_iterator.center())
        face_id = target_face_iterator.index()

        # print(face_id, face_mean.x, face_mean.y, face_mean.z)

        # TODO Send rays in all directions from the currect vertex
        # Random? Uniform? Same for all vertices? 
        # TODO define depth of testing -- length of the vectors
        rayDir = OpenMaya.MFloatVector(150., 0., 0.)

        # search setup (for all cases)
        if not test_intersect(camera_surface_mesh, face_mean, rayDir, cam_surface_accelerator):  # no intesection with camera surface
            invisible_counter += 1
            target_mesh.deleteFace(face_id)
            last_deleted = face_id
            to_delete_list.append(face_id)
        if test_intersect(target_mesh, face_mean, rayDir, target_accelerator, at_least_2=False):  # intersects itself
            self_intersect += 1
            if last_deleted != face_id:
                target_mesh.deleteFace(face_id)
                last_deleted = face_id
                to_delete_list.append(face_id)
        if any([test_intersect(mesh, face_mean, rayDir, acc) for mesh, acc in zip(obstacles_meshes, obstacles_accs)]):  # intesects any of the obstacles
            # vertex is invisible from camera
            # print('Vertex {}: {} is not visible.'.format(vertex_id, raySource))
            object_intersect += 1
            if last_deleted != face_id:
                target_mesh.deleteFace(face_id)
                last_deleted = face_id
                to_delete_list.append(face_id)
        
        target_face_iterator.next()  # iterate!

    # Remove invisible vertices
    print('{} invisible, {} self-intersects, {} obstacle instersects out of {}'.format(invisible_counter, self_intersect, object_intersect, target_mesh.numPolygons()))

    print('To delete {} : {}'.format(len(to_delete_list), to_delete_list))

    # TODO use MDGModifier instead
    # to_delete.doIt()
    # for face_id in to_delete:
    #     target_mesh.deleteFace(face_id)
    #     print('Removed {}'.format(face_id))


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