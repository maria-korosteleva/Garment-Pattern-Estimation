# How to run NeuralTailor

Don't forget to follow the installation steps first: [Installation](Installation.md)

> ☝ **NOTE**: All the config files specify local paths relative to the root, so we recommend running all the commands from the root of the code directory to avoid path problems.

## Evaluation

`nn/evaluation_scripts/` contains several tools that performs evaluations of the trained models. Run the script of your intrest with `--help` option to get the information about parameters.

`maya_att_weights.py` is the exception, this is just a helper script to visualize attention weights predicted by a model (they are saved alongside the sewing pattern predictions) within Autodesk Maya environment


### Saved models VS Weights & Biases runs

Every evaluation scrips takes in a config file which describes the experiment to evaluate. The scripts can work with either models saved locally or W&B runs.

1. Locally saved models. We provide pre-trained NeuralTailor models (patter shape prediction and stitch model prediction) in `./models/` folder. Corresponding configuration files (e.g., `./models/att/att.yaml`) contain full information about hyperparameters, dataset, and paths to the pre-trained model weights. You can similarly create configuration files to work with locally saved models produced by your experiments.

2. Weights&Biases runs (easier for your trained models). When training a framework, all the experiment information is logged to W&B cloud. Evaluation scripts can work with those runs directly without a need to manually download models and fill configurations. 

    To run scripts with W&B runs simply provide the related info of project name, runs name and run id in the `experiment` section of configuration file, and specify `unseen_data_folders` of `dataset` section if evaluating on unseen garment types. What's specified in the rest of the config is irrelevant since it will overriden by the information from the cloud run. Here is an [example of such evaluation config](../nn/example_configs/eval_wandb.yaml).

### Tweaking evaluation parameters


`/nn/evaluation_scripts/on_test_set.py` allows updating some parameters of the dataset for evaluation purposes, e.g. add point cloud noise or evaluate on scan imitation version of the input garments.
To do so, specify new values in the `load_dataset(..)` function calls in the script. The script itself contains some examples.


### Examples of evaluation commands

Evaluate NeuralTailor pattern shape prediction model on seen garment types only:

```
python nn/evaluation_scripts/on_test_set.py -sh models/att/att.yaml
```

> NOTE: when evaluating only the pattern shape model without stitches, the stitches are transferred from the corresponding GT sewing patterns (if available) for convenience of loading and draping. 

Evaluate full NeuralTailor framework on unseen garment types and save sewing pattern predictions:

```
python nn/evaluation_scripts/on_test_set.py -sh models/att/att.yaml -st models/att/stitch_model.yaml --unseen --predict
```

Evaluate stitch model on previously saved sewing pattern predictions: 

```
python nn/evaluation_scripts/on_test_set.py -st models/att/stitch_model.yaml --pred_path /path/to/sewing/pathern/data 
```


## Training

The training of NeuralTailor is two-step -- separately for Pattern Shape and Stitch Information models. 
You can use config files saved in `models/` as training configs.

1. Pattern Shape Regression Network training
    To run with our final NeuralTailor architecture setup simply run this command from the directory root: 
    ```
    python nn/train.py -c ./models/att/att.yaml
    ```
    Training on the full dataset will take 2-4 days depending on your hardware. 
2. Stitch training 
    * Runs after the Shape Regression Network
    * Update the name & id of your shape training run in the Stitch model config file, 'old_experiment' section. Setting this option enables training on the Pattern Shape predictions. Put the 'old_experiment' -> 'predictions' to False, or removing the 'old_experiment' section altogehter will result in training on GT sewing patterns.
    * Run: 
    ```
    python nn/train.py -c ./models/att/stitch_model.yaml
    ```

### Reproducing other experiments reported in the paper

By modifying the configuration files for corresponding models one could reproduce the setups used in our reported experiments. Some examples:
* setting `dataset->filter_by_params` option to empty string or null will force the training process to use full dataset without filtering out the desing overlaps.
* changing the model class name (`NN->model`) to `GarmentFullPattern3D` will result in training of LSTM-based model with global latent space and LSTM-based pattern decoder (our baseline)
* adding `stitch, free_class` to `loss_components` and `quality_components` will enable training a model that predicts stitches using stitch tags as part of sewing pattern shape model.
* changing `dataset->panel_classification` to `./nn/data_configs/panel_classes_plus_one.json` will give you a run with alternative panel classes arrangement.

>**NOTE:** if the config changes are expected to affect the list of the datapoints used for training (changing `filter_by_params` or `max_datapoints_per_type`), the provided data splits into train\valid\test might become invalid. Remove `data_split->filename` to allow training process to create new split on the go. We only provide splits for dataset with and without parameter filtering (in `nn/data_configs`).

### Offline training

By default it sycronises the training run information with (Weights&Biases)[wandb.ai] cloud. To disable this sycronization (run offline), set your environemtal variable: 

```
WANDB_MODE="offline"
```
Source: [W&B Documentation](https://docs.wandb.ai/guides/track/launch#is-it-possible-to-save-metrics-offline-and-sync-them-to-w-and-b-later)

> **NOTE:** All secondary scripts (eveluation, stitch training) will require setting up configs for using locally saved models (as described above) to evaluate on these offline runs. 