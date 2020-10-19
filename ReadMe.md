# Estimating 2D pattern from a 3D garment

## Licensing 
This is work-in-progress, and licensing terms are not finalized. If you want to use this code, contact the authors. 

## Filesystem paths settings
Create system.json file in the root of this directory with your machine's file paths using system.template.json as a template. 
system.json should include the following: 
* General
    * path for creating logs at (including generated data from dataset generation routined & NN predictions) ('output')

* Data generation & Simulation resources 
    * path to pattern templates folder ('templates_path') 
    * path to folder containing body files ('bodies_path')
    * path to folder containing rendering setup (scenes) ('scenes_path')
    * path to folder with simulation\rendering configurations ('sim_configs_path')

* NN Training references
    * path to finalized garment datasets that could be used for training NN ('datasets_path')
    * username for wandb tool for correct experiment tracking ('wandb_username')

## Data Creation

Provides tools for creating datasets of 3D garments with patterns. See [Data Creation Readme](data_generation/ReadMe.md).

## Neural Network

Provides tools to train and run the pattern prediction model on 3D garment geometry 

## Dependencies

### Notes on errors with PIL.Image

You might experience errors related with PIL (pillow) Image module. Those most often come from the ReportLab library requiring older versions of pillow that are currently available, the compatibility issues of pillow and python version, or ReportLab and libigl visualization routines requiring different versions of pillow

*Working combinations*:
* For ReportLab (saving patterns as png images) to work: Python 3.8.5 + ReportLab 3.5.53 + pillow 7.1.1

### Main Dependencies

All development was done on Windows 10. If running on other OS endups up with errors, please, raise the issue!

* Python 3.6
* Numpy
* Pytorch
* torch-geometric (https://pytorch-geometric.readthedocs.io/en/latest/index.html)
* scipy
* Maya
* Qualoth for Maya
* igl (Python, https://libigl.github.io/libigl-python-bindings/) + meshplot for viewing