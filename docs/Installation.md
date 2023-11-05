# Installation & Dependencies

This file describes the process of setting up the environment from scratch (with the working versions). Skip to the relevant sections as needed

## 1. Dataset

* Download the [Dataset of 3D Garments with Sewing Patterns](https://zenodo.org/record/5267549#.Yk__mMgzaUk) in order to train\evaluate NeuralTailor.
    > NOTE: For evaluation of pre-trained NeuralTailor on unseen types you only need the _test.zip_ part of the dataset. 

* Unpack all ZIP archives to the same directory, keeping the directory structure (every zip archive is a subfolder of your root). 

## 2. Basic environment: Miniconda
```
apt-get update && apt-get install -y wget

# conda: https://stackoverflow.com/questions/58269375/how-to-install-packages-with-miniconda-in-dockerfile
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b

# Env variable 
export PATH="$HOME/miniconda3/bin:$PATH"

# Create enviroment
conda create -n Garments python=3.9
conda activate Garments
```

## 3. Dependencies

* Pytoch (with cudatools if cuda is available on the machine)

* (PyG)[https://pytorch-geometric.readthedocs.io/en/latest/] for graph layers. Note that installing with pip requires specifying installed versions of pytorch and cuda. The installation may not support all possible combinations.

* [libigl](https://github.com/libigl/libigl-python-bindings) needs installation with conda. You could also check other options on [their GitHub page](https://github.com/libigl/libigl-python-bindings)

* The rest of the requirements are provided in `requirements.txt`. They can be used with pip to install in one go: 
    ```
    pip install -r requirements.txt
    ```

Example set of commands for installation on Windows [the version are checked and should work]:

```
conda create -n Garments python=3.9
conda activate Garments

conda install pytorch==1.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

conda install -c conda-forge igl=2.2.1

conda install pywin32   # requirement for wmi on Windows -- conda installation needed in conda environements
# The rest are in requirements.txt
pip install -r requirements.txt

```

Example set of commands for installation on Linux [Ubuntu]:
```
conda create -n Garments python=3.9
conda activate Garments

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# for pytorch-cuda, insert cuda version that is listed under nvidia-smi
# in this case, the cuda version is 12.1

conda install pyg -c pyg
conda install pytorch-cluster -c pyg

conda install -c conda-forge igl=2.2.1 svgwrite svglib wandb

pip install sparsemax entmax
```

Development was done on _Windows 10\11_ and Ubuntu. If running on other OS ends up with errors, please, raise an issue!

**Notes on errors with PIL.Image**

You might experience errors related with PIL (pillow) Image module. Those most often come from the ReportLab library requiring older versions of pillow that are currently available, the compatibility issues of pillow and python version, or ReportLab and libigl visualization routines requiring different versions of pillow. 

*Working combinations*:
* For ReportLab (saving patterns as png images) to work: 
    * Python 3.8.5 + ReportLab 3.5.53 + pillow 7.1.1
    * Python 3.8.5 + ReportLab 3.5.55 + pillow 7.1.1

## 4. (Optional) Weights & Biases account

We are using [Weights & Biases](https://wandb.ai/) for experiment tracking. 

You can use evalution scripts on provided models or train new models without having your own W&B account, but we recommend to create one -- then all the information of your training runs will be fully private and will be stored in your account forever. Anonymous runs are only retained for 7 days (as of April 2022).

The prompt to authenticate will appear the first time you run any of the scripts that use w&b library.

> NOTE: Anonimous runs are not yet supported by our tool. 


## 5. Custom dependencies access

Download [Garment-Pattern-Generator](https://github.com/maria-korosteleva/Garment-Pattern-Generator) code (for pattern loading)

```
git clone https://github.com/maria-korosteleva/Garment-Pattern-Generator
```

Add path to custom packages to PYTHONPATH for correct importing of our custom modules. For example, in the terminal:
```
export PYTHONPATH=$PYTHONPATH:/home/user/maria/Garment-Pattern-Generator/packages
```

### Filesystem paths & W&B account settings
* Fill out system.json file
Create system.json file in the root of this directory with your machine's file paths using system.template.json as a template. 
system.json should include the following: 
* path for creating logs at (including generated data from dataset generation routined & NN predictions) ('output')
    ```
    mkdir outputs
    ```
* path to the root directory with downloaded and unpacked per-type garment datasets to be used for training\evaluating of NN ('datasets_path') 
* username for wandb for correct experiment tracking ('wandb_username'). This is optional for evaluating saved models or running training (training will fall into anonymous mode). However, if you were using anonymous mode for training and want to use evaluation scripts with that run, please specify the temporary account name (printed when training and can be found in the run URL) in this field for correct URL construction.

