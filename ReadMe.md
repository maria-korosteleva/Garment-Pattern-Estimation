# Estimating 2D pattern from a 3D garment

## Licensing 
This is work-in-progress, and licensing terms are not finalized. If you want to use this code, contact the authors. 

## Filesystem paths settings
Create system.json file in the root of this directory with your machine's file paths using system.template.json as a template. 
system.json should include the following: 
### General
* path for creating logs & putting new datasets to ('output')

### Data generation & Simulation resources 
* path to pattern templates folder ('templates_path') 
* path to folder containing body files ('bodies_path')
* path to folder containing rendering setup (scenes) ('scenes_path')
* path to folder with simulation\rendering configurations ('sim_configs_path')

## Data Creation

Provides tools for creating datasets of 3D garments with patterns. See [Data Creation Readme](data_generation/ReadMe.md).

## Neural Network

Provides tools to train and run the pattern prediction model on 3D garment geometry 

### Main Dependencies
* Python 3.6
* Numpy
* Pytorch
* scipy
* Maya
* Qualoth for Maya
* igl (Python, https://libigl.github.io/libigl-python-bindings/) + meshplot for viewing