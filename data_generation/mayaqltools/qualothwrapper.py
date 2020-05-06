"""
    Qualoth scripts are written in MEL. 
    This module makes a python interface to them
    Notes:
        * this module is Python 2.7-friendly
        * Error checks are sparse to save coding time & lines. 
            This sould not be a problem during the normal workflow
    
"""
from __future__ import print_function
import time

from maya import mel
from maya import cmds


def load_plugin():
    """
        Forces loading Qualoth plugin into Maya. 
        Note that plugin should be installed and licensed to use it!
        Inquire here: http://www.fxgear.net/vfxpricing
    """
    maya_year = int(mel.eval('getApplicationVersionAsFloat'))
    plugin_name = 'qualoth_' + str(maya_year) + '_x64'
    print('Loading ', plugin_name)

    cmds.loadPlugin(plugin_name)


# -------- Wrappers -----------
# Make sure that Qualoth plugin is loaded before running any wrappers!

def qlCreatePattern(curves_group):
    """
        Converts given 2D closed curve to a flat geometry piece
    """
    objects_before = cmds.ls(assemblies=True)
    # run
    cmds.select(curves_group)
    mel.eval('qlCreatePattern()')
    
    # Identify newly created objects
    objects_after = cmds.ls(assemblies=True)
    # No need for symmetric difference because we don't care if some objects were deleted
    return list(set(objects_after) - set(objects_before))


def qlCreateSeam(curve1, curve2):
    """
        Create a seam between two selected curves
        TODO add support for 1-many stitches
    """
    cmds.select([curve1, curve2])
    # Operates on selection
    seam_shape = mel.eval('qlCreateSeam()')
    return seam_shape


def qlCreateCollider(cloth, target):
    """
        Marks object as a collider object for cloth --
        eshures that cloth won't penetrate body when simulated
    """
    objects_before = cmds.ls(assemblies=True)

    cmds.select([cloth, target])
    # Operates on selection
    mel.eval('qlCreateCollider()')

    objects_after = cmds.ls(assemblies=True)
    return list(set(objects_after) - set(objects_before))


# ------- Higher-level functions --------

def start_maya_sim(garment, props):
    """Start simulation through Maya defalut playback without checks
        Gives Maya user default control over stopping & resuming sim
        Current qlCloth material properties from Maya are used (instead of garment config)
    """
    config = props['config']
    solver = _init_sim(config)

    # Allow to assemble without gravity
    print('Simulation::Assemble without gravity')
    _set_gravity(solver, 0)
    for frame in range(1, config['zero_gravity_steps']):
        cmds.currentTime(frame)  # step

    # resume normally
    print('Simulation::normal playback.. Use ESC key to stop simulation')
    _set_gravity(solver, -980)
    cmds.currentTime(frame - 1)  # one step back to start from simulated state
    cmds.play()


def run_sim(garment, props):
    """
        Setup and run cloth simulator untill static equlibrium is achieved.
        Note:
            * Assumes garment is already properly aligned!
            * All of the garments existing in Maya scene will be simulated
                because solver is shared!!
    """
    config = props['config']
    solver = _init_sim(config)
    garment.setMaterialSimProps(config['material'])  # ensure running sim with suplied material props
    
    start_time = time.time()
    # Allow to assemble without gravity + skip checks for first few frames
    _set_gravity(solver, 0)
    for frame in range(1, config['zero_gravity_steps']):
        cmds.currentTime(frame)  # step
        garment.cache_if_enabled(frame)

    # resume normally
    _set_gravity(solver, -980)
    for frame in range(config['zero_gravity_steps'], config['max_sim_steps']):
        cmds.currentTime(frame)  # step
        garment.cache_if_enabled(frame)
        garment.update_verts_info()
        if garment.is_static(config['static_threshold']):  
            # TODO Add penetration checks
            # Success!
            break

    # Fail check: static equilibrium never detected -- might have false negs!
    if frame == config['max_sim_steps'] - 1:
        props['stats']['sim_fails'].append(garment.name)

    # TODO make recording pattern-specific, not dataset-specific
    props['stats']['sim_time'].append(time.time() - start_time)
    props['stats']['spf'].append(props['stats']['sim_time'][-1] / frame)
    props['stats']['fin_frame'].append(frame)


def findSolver():
    """
        Returns the name of the qlSover existing in the scene
        (usully solver is created once per scene)
    """
    solver = cmds.ls('*qlSolver*Shape*')
    return solver[0]


def deleteSolver():
    """deletes all solver objects from the scene"""
    cmds.delete(cmds.ls('*qlSolver*'))


def setColliderFriction(collider_objects, friction_value):
    """Sets the level of friction of the given collider to friction_value"""

    main_collider = [obj for obj in collider_objects if 'Offset' not in obj]
    collider_shape = cmds.listRelatives(main_collider[0], shapes=True)

    cmds.setAttr(collider_shape[0] + '.friction', friction_value)


def setMaterialProps(cloth, props):
    """Set given material propertied to qlClothObject"""
    if not props:
        return
    # Simple ones
    cmds.setAttr(cloth + '.density', props['density'], clamp=True)  
    cmds.setAttr(cloth + '.stretch', props['stretch_resistance'], clamp=True)
    cmds.setAttr(cloth + '.shear', props['shear_resistance'], clamp=True)
    cmds.setAttr(cloth + '.stretchDamp', props['stretch_damp'], clamp=True)
    cmds.setAttr(cloth + '.bend', props['bend_resistance'], clamp=True)
    cmds.setAttr(cloth + '.bendAngleDropOff', props['bend_angle_dropoff'], clamp=True)
    cmds.setAttr(cloth + '.bendDamp', props['bend_damp'], clamp=True)
    cmds.setAttr(cloth + '.bendDampDropOff', props['bend_damp_dropoff'], clamp=True)
    cmds.setAttr(cloth + '.bendYield', props['bend_yield'], clamp=True)
    cmds.setAttr(cloth + '.bendPlasticity', props['bend_plasticity'], clamp=True)
    cmds.setAttr(cloth + '.viscousDamp', props['viscous_damp'], clamp=True)
    cmds.setAttr(cloth + '.friction', props['friction'], clamp=True)
    cmds.setAttr(cloth + '.pressure', props['pressure'], clamp=True)
    cmds.setAttr(cloth + '.lengthScale', props['length_scale'], clamp=True)
    cmds.setAttr(cloth + '.airDrag', props['air_drag'], clamp=True)
    cmds.setAttr(cloth + '.rubber', props['rubber'], clamp=True)

    # need setting flags
    cmds.setAttr(cloth + '.overrideCompression', 1)
    cmds.setAttr(cloth + '.compression', props['compression_resistance'], clamp=True)

    cmds.setAttr(cloth + '.anisotropicControl', 1)
    cmds.setAttr(cloth + '.uStretchScale', props['weft_resistance_scale'], clamp=True)
    cmds.setAttr(cloth + '.vStretchScale', props['warp_resistance_scale'], clamp=True)
    cmds.setAttr(cloth + '.rubberU', props['weft_rubber_scale'], clamp=True)
    cmds.setAttr(cloth + '.rubberV', props['warp_rubber_scale'], clamp=True)


def fetchMaterialProps(cloth):
    """Returns current material properties of the cloth's objects
        Requires qlCloth object
    """
    props = {}
    # Mass density per unit area. (Kg/cm2)
    props['density'] = cmds.getAttr(cloth + '.density')  
    # Resisting force to planar stretching and compression
    props['stretch_resistance'] = cmds.getAttr(cloth + '.stretch') 
    # Resisting force to shearing. (See Figure.) This parameter is 
    # interpreted as a scale factor to the stretch resistance. 
    props['shear_resistance'] = cmds.getAttr(cloth + '.shear')
    # Damping factor for stretching motion.
    props['stretch_damp'] = cmds.getAttr(cloth + '.stretchDamp')
    # Resisting force to bending.
    props['bend_resistance'] = cmds.getAttr(cloth + '.bend')
    props['bend_angle_dropoff'] = cmds.getAttr(cloth + '.bendAngleDropOff')
    # Damping factor for bending motion
    props['bend_damp'] = cmds.getAttr(cloth + '.bendDamp')
    props['bend_damp_dropoff'] = cmds.getAttr(cloth + '.bendDampDropOff')

    # creases: elasticity vs plasticity
    props['bend_yield'] = cmds.getAttr(cloth + '.bendYield')
    props['bend_plasticity'] = cmds.getAttr(cloth + '.bendPlasticity')

    # external
    # This damping force drags the motion of each cloth vertex in all directions uniformly 
    # regardless of the directions of normals of each vertex.
    props['viscous_damp'] = cmds.getAttr(cloth + '.viscousDamp')
    # Controls the friction among cloth objects or colliders. Also self-friction
    props['friction'] = cmds.getAttr(cloth + '.friction')
    # The amount of pressure force which are applied to the vertex normal directions of each cloth vertex.
    props['pressure'] = cmds.getAttr(cloth + '.pressure')

    # need setting flags
    # need to turn on .overrideCompression
    props['compression_resistance'] = cmds.getAttr(cloth + '.compression')

    # ------ unlikely to be used ---------
    # Scale factor for length unit. 
    props['length_scale'] = cmds.getAttr(cloth + '.lengthScale') 
    # value controls the amount of influence from those air fields. In case 
    # here is no attached field to this cloth, 'Air Drag' simply drags the 
    # cloth motion in the direction of face normals of each triangle. 
    props['air_drag'] = cmds.getAttr(cloth + '.airDrag')
    # This value scales the area of the cloth in rest state. 
    props['rubber'] = cmds.getAttr(cloth + '.rubber')
    # fine-grained
    # The scale factor to the planar stretching/compression resistance in weft (U) direction.
    props['weft_resistance_scale'] = cmds.getAttr(cloth + '.uStretchScale')
    # The scale factor to the planar stretching/compression resistance in warp (V) direction.
    props['warp_resistance_scale'] = cmds.getAttr(cloth + '.vStretchScale')
    # The scale factor to the rubber value (rest length scale) in weft (U) direction.
    props['weft_rubber_scale'] = cmds.getAttr(cloth + '.rubberU')
    # The scale factor to the rubber value (rest length scale) in warp (V) direction.
    props['warp_rubber_scale'] = cmds.getAttr(cloth + '.rubberV')

    return props


# ------- Utils ---------
def _init_sim(config):
    """
        Basic simulation settings before starting simulation
    """
    solver = findSolver()

    cmds.setAttr(solver + '.selfCollision', 1)
    cmds.setAttr(solver + '.startTime', 1)
    cmds.setAttr(solver + '.solverStatistics', 0)  # for easy reading of console output
    cmds.playbackOptions(ps=0, max=config['max_sim_steps'])  # 0 playback speed = play every frame

    return solver


def _set_gravity(solver, gravity):
    """Set a given value of gravity to sim solver"""
    cmds.setAttr(solver + '.gravity1', gravity)