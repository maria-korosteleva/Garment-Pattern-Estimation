from PythonQt import QtCore, QtGui, MarvelousDesignerAPI
from PythonQt.MarvelousDesignerAPI import *
import MarvelousDesigner
from MarvelousDesigner import *

#If you want to load your python file, please type "sys.path.append(input path where file is located here)" in console.
#ex) sys.path.append("C:/Users/Young/Downloads/") or sys.path.append("C:\\\\Users\\\\Young\\\\Downloads\\\\")
class example():
    #this function is an exmaple of single process
    #single process defines a series of actions, which is 'set an option' ->'set a path to be loaded'->'set a simulation option'->'set a save path'->'call a processing function'.
    #You can designate the path for each file you want to load.
    #Also You can designate a save path for each file
    #object is mdsa (MarvelousDesigner Script API)
    def run_single_process_example(self, object): 
        # clear console window
        object.clear_console() 
        #initialize mdsa module
        object.initialize() 
        #set exporting option
        object.set_open_option("mm", 30) 
        #set importing option
        object.set_save_option("cm", 30, False)
        #Set the path of an Avatar file you want to load.
        object.set_avatar_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Avatar\\Avatar\\Female_A_V3.avt")
        #Set the path of a Garment file you want to load.
        object.set_garment_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Garment\\default.zpac") 
        #Set the path of an Animation file you want to load.
        object.set_animation_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Avatar\\Pose\\Female_A\\F_A_pose_03.pos") 
        #Set simulation option.
        object.set_simulation_options(0, 0, 5000) 
        #Set the saving file path.
        object.set_save_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\test_04.obj")
        #If you finish setting file paths and options. You must call process function.
        object.process(True) 

    #this function is anohter exmaple for single process 
    #call 'single_process' function with the each file path, options and paths to save file
    #also designate the folder where the files will be stored.
    def run_single_process_second_example(self, object):
        # clear console window
        object.clear_console() 
        #initialize mdsa module
        object.initialize() 
        #set exporting option
        object.set_open_option("cm", 30) 
        #set importing option
        object.set_save_option("mm", 30, False) 
        #designate the folder where the files will be stored and file extension setting
        object.set_save_folder_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999", "mc") 
        #call the "single_process" function
        object.single_process(
            "C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Garment\\default.zpac",
            "C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Avatar\\Avatar\\Female_A_V3.avt",
            "C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Avatar\\Pose\\Female_A\\F_A_pose_02_attention.pos") 

    #this function is exmaple for multi process 
    def run_multi_example(self, object):
        # clear console window
        object.clear_console() 
        #initialize mdsa module
        object.initialize()
        #Set importing options (unit) of list type
        import_unit_list = ["mm", "mm", "mm", "mm"]
        #Set importing options (fps) of list type
        import_fps_list = [30, 24, 30, 60]
        #Set exporting options (unit) of list type
        export_unit_list = ["cm", "cm", "cm", "cm"]
        #Set exporting options (fps) of list type
        export_fps_list = [30, 24, 30, 60]
        #Set exporting options (set a unified map) of list type
        export_unified_map_list = [False, False, True, True]
        #call 'set_open_option_list' with variable of list type (import_unit_list, import_fps_list)
        object.set_open_option_list(import_unit_list, import_fps_list)
        #call 'set_save_option_list' with variable of list type (export_unit_list, export_fps_list, export_unified_map_list)
        object.set_save_option_list(export_unit_list, export_fps_list, export_unified_map_list)
        #designate the folder where the garment files to load are located and file extension
        object.set_garment_folder("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Garment", "zpac")
        #designate the folder where the avatar files to load are located and file extension 
        #object.set_avatar_folder("C:\\\\Users\\Public\\\\Documents\\\\MarvelousDesigner\\\\Assets_ver_3.0.9999\\\\Garment", "zpac")
        #designate the folder where the animation files to load are located and file extension
        object.set_animation_folder("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\alembic", "abc")
        #designate the folder where the files will be stored and file extension
        object.set_save_folder_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999", "fbx")
        #call the "process" function
        object.process(True)

    #this function is example for multi process 
    def run_multi_second_example(self, object):
        # clear console window
        object.clear_console() 
        #initialize mdsa module
        object.initialize()
        #In case want to simulate/record one garment and avatar with multiple animation
        #set path of one garment file
        object.set_garment_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Garment\\default.zpac")
        #set path of one avatar file
        object.set_avatar_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Avatar\\Avatar\\Female_A_V3.avt")
        #set folder path of multiple animation folder and extension (file extension must be supported by Marvelous Designer)
        object.set_animation_folder("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Avatar\\Pose\\Female_A", "pos")
        #set save folder and extension (file extension must be supported by Marvelous Designer)
        object.set_save_folder_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999", "abc")
        #call the "process" function (to autosave project file, change factor to ture)
        object.process(False)

    #this function is example for serial multiple process  
    def run_multi_third_example(self, object):
        # clear console window
        object.clear_console() 
        #initialize mdsa module
        object.initialize()
        #Set importing options (unit) of string type
        object.import_unit = "cm"
        #Set importing options (fps) of integer type
        object.import_fps = 30
        #Set exporting options (unit) of string type
        object.export_unit = "cm"
        #Set exporting options (fps) of integer type
        object.export_fps = 30
        #Set exporting options (set a unified map) of boolean type
        object.export_unified_map = False
        #Set exporting options (garment thin / thick Mode) of boolean type
        object.export_thin = False
        #Set exporting option (weld / unweld) of boolean type
        object.export_weld = True

        #set path of one garment file
        object.set_garment_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Garment\\default.zpac")
        #set path of one avatar file
        #alembic does not need an avatar file.
        #If you use a different format, delete "#"
        #object.set_avatar_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Avatar\\Avatar\\Female_A_V3.avt")
        #set folder path of multiple animation folder and extension (file extension must be supported by Marvelous Designer)
        
        object.set_animation_folder("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\pose01", "abc")
        #must call object.sync_file_lists for serial multiple process
        object.sync_file_lists("animation")
        #next multi process
        object.set_garment_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Garment\\basic_pants.zpac")
        #object.set_avatar_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Avatar\\Avatar\\Female_A_V3.avt")
        object.set_animation_folder("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\pose02", "abc")
        object.sync_file_lists("animation")
        #next multi process
        object.set_garment_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Garment\\T-shirt_men.zpac")
        #object.set_avatar_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Avatar\\Avatar\\Female_A_V3.avt")
        object.set_animation_folder("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\pose03", "abc")
        object.sync_file_lists("animation")
        #next multi process
        object.set_garment_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Garment\\T-shirt_men.zpac")
        #object.set_avatar_file_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\Avatar\\Avatar\\Female_A_V3.avt")
        object.set_animation_folder("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999\\pose04", "abc")
        object.sync_file_lists("animation")
        #setting file path for serial multiple process is done, designate path for save and extension
        object.set_save_folder_path("C:\\Users\\Public\\Documents\\MarvelousDesigner\\Assets_ver_3.0.9999", "mc")
        #Call the "process" function
        object.process(False)




