"""Shares utils to work with Maya"""

import ctypes
from maya import OpenMaya
from maya import cmds

# ----- Working with Mesh objects -----

def get_dag(object_name):
    """Return DAG for requested object"""
    selectionList = OpenMaya.MSelectionList()
    selectionList.add(object_name)
    dag = OpenMaya.MDagPath()
    selectionList.getDagPath(0, dag)
    return dag


def get_mesh_dag(object_name):
    """Return MFnMesh object by the object name"""
    # get object as OpenMaya object -- though DAG
    dag = get_dag(object_name)
    # as mesh
    mesh = OpenMaya.MFnMesh(dag)  # reference https://help.autodesk.com/view/MAYAUL/2017/ENU/?guid=__py_ref_class_open_maya_1_1_m_fn_mesh_html

    return mesh, dag


def test_ray_intersect(mesh, raySource, rayVector, accelerator=None, hit_tol=None, return_info=False):
    """Check if given ray intersect given mesh
        * hit_tol ignores intersections that are within hit_tol from the ray source (as % of ray length) -- usefull when checking self-intersect
        * mesh is expected to be of MFnMesh type
        * accelrator is a stucture for speeding-up calculations.
            It can be initialized from MFnMesh object and should be supplied with every call to this function
    """
     # It turns out that OpenMaya python reference has nothing to do with reality of passing argument:
    # most of the functions I use below are to be treated as wrappers of c++ API
    # https://help.autodesk.com/view/MAYAUL/2018//ENU/?guid=__cpp_ref_class_m_fn_mesh_html

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
        hit = any([dist > hit_tol for dist in hitRayParams])

    if return_info:
        return hit, hitFaces, hitPoints, hitRayParams
    
    return hit


def edge_vert_ids(mesh, edge_id):
    """Return vertex ids for a given edge in given mesh"""
    # Have to go through the C++ wrappers
    # Vertices that comprise an edge
    script_util = OpenMaya.MScriptUtil(0.0)
    v_ids_cptr = script_util.asInt2Ptr()  # https://forums.cgsociety.org/t/mfnmesh-getedgevertices-error-on-2011/1652362
    mesh.getEdgeVertices(edge_id, v_ids_cptr) 

    # get values from SWIG pointer https://stackoverflow.com/questions/39344039/python-cast-swigpythonobject-to-python-object
    ty = ctypes.c_uint * 2
    v_ids_list = ty.from_address(int(v_ids_cptr))
    return v_ids_list[0], v_ids_list[1]


def save_mesh(target, to_file):
    """Save given object to file as a mesh"""

    # Make sure to only select requested mesh
    cmds.select(clear=True)
    cmds.select(target)

    cmds.file(
        to_file,
        type='OBJExport',  
        exportSelectedStrict=True,  # export selected -- only explicitely selected
        options='groups=0;ptgroups=0;materials=0;smoothing=0;normals=1',  # very simple obj
        force=True,   # force override if file exists
        defaultExtensions=False
    )

    cmds.select(clear=True)
