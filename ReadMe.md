# Estimating 2D pattern from a 3D garment

## Licensing 
This is work-in-progress, and licensing terms are not finalized. If you want to use this code, contact the authors. 

## Modules
---

### Data Creation

Provides tools for creating datasets of 3D garments with patterns. See [Data Creation Readme](data_generation/ReadMe.md).

### Training and evaluating Neural Network to predict pattern from 3D garment model

Provides tools to train and run the pattern prediction model on 3D garment geometry 

## Filesystem paths settings
---
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



## Dependencies
---

* Developed and tested on Python 3.6+
* For libs, `requirements.txt` is given for a refence, but due to the specifics of some packages, using `pip install -r requirements.txt` will likely fail
* When planning to use [PyTorch] with CUDA enables, it's recommended to install it using `conda` to explicitely install cudatools: 
    ```
    conda install pytorch cudatoolkit -c pytorch
    ```
    * This repo was tested with PyTorch 1.6-1.7 and CUDA 10.1, 10.2, 11.0
* [libigl](https://github.com/libigl/libigl-python-bindings) needs installation with conda. You could also check other options on [their GitHub page](https://github.com/libigl/libigl-python-bindings)
    ```
    conda install -c conda-forge igl
    ```
* [torch-geometric installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) might also require additional attention as the versions of dependencies depend on the CUDA version PyTorch is using on your machine
* Simple installation with `pip install <package_name>`: 
    * `wandb` 
    * `svglib` 
    * `svgwrite`
* The project has custom Python packages. Add `Garment-Pattern-Estimation/packages` to `PYTHONPATH` environmental variable for the project to work correctly!

Development was done on _Windows 10_. If running on other OS ends up with errors, please, raise an issue!

**Notes on errors with PIL.Image**

You might experience errors related with PIL (pillow) Image module. Those most often come from the ReportLab library requiring older versions of pillow that are currently available, the compatibility issues of pillow and python version, or ReportLab and libigl visualization routines requiring different versions of pillow. 

*Working combinations*:
* For ReportLab (saving patterns as png images) to work: 
    * Python 3.8.5 + ReportLab 3.5.53 + pillow 7.1.1
    * Python 3.8.5 + ReportLab 3.5.55 + pillow 7.1.1

### Additional requirements for Simulating garment patterns on a human 3D model

* Autodesk Maya 2018 or Autodesk Maya 2020 
    * With `numpy` installed (there are a number of instruction around the Internet on how to get `numpy` to work in Maya)
* [Qualoth](https://www.qualoth.com/) for Maya as Simulator

See more details in dedicated [Data Creation Readme](data_generation/ReadMe.md).