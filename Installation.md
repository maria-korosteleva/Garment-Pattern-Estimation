# Installation & Dependencies

This file describes the process of setting up the environment from scratch (with the working versions). Skip to the relevant sections as needed

## 1. Basic environment: Miniconda
```
apt-get update && apt-get install -y wget

# conda: https://stackoverflow.com/questions/58269375/how-to-install-packages-with-miniconda-in-dockerfile
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b

# Env variable 
export PATH="$HOME/miniconda3/bin:$PATH"

# Create enviroment
conda create -n Garments python=3.8.5
conda activate Garments
```

## 2. Dependencies

```
conda install pytorch=1.9.1 cudatoolkit=10.2 -c pytorch

conda install -c conda-forge igl=2.2.1

git clone https://github.com/subhadarship/kmeans_pytorch && cd kmeans_pytorch && git checkout a65871651e9b38f89fa2bf0b02c0170bf40b52bf && pip install --editable .

# The rest are in requirements.txt
pip install -r requirements.txt

```

## 3. Custom dependencies access

```
git clone <data_gen>
```

Add path to custom packages to PYTHONPATH, for example, in the terminal:
```
export PYTHONPATH=$PYTHONPATH:/home/user/maria/Garment-Pattern-Data/packages
```


### Filesystem paths settings
* Fill out system.json file
Create system.json file in the root of this directory with your machine's file paths using system.template.json as a template. 
system.json should include the following: 
* path for creating logs at (including generated data from dataset generation routined & NN predictions) ('output')
    ```
    mkdir outputs
    ```
* path to finalized garment datasets that could be used for training NN ('datasets_path')
* username for wandb tool for correct experiment tracking ('wandb_username')
