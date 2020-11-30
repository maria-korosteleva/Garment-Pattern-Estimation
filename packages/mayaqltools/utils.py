"""Shares utils to work with surfaces in Maya"""
from maya import OpenMaya


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