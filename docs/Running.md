# How to run NeuralTailor

Don't forget to follow the installation steps first: [Installation](Installation.md)

## Training

The training of NeuralTailor is two-step -- separately for Pattern Shape and Stitch Information models. 
The training setup for both parts are given in [nn/train_configs/](../nn/train_configs/)

1. Pattern Shape Regression Network training
    To run with default setup simply run this command from the directory root : 
    ```
    python nn/train.py -c ./nn/train_configs/att.yaml
    ```
    > Training on the full dataset will take 2-4 days depending on your hardware. 
2. Stitch training 
    * Runs after the Shape Regression Network
    * Update the name & id of the shape training run in the [Stitch model config file](../nn/train_configs/stitch_model.yaml), 'old_experiment' section. Setting this option enables training on the Pattern Shape predictions. 
        To train the stitch model on Ground Truth panel shapes, simply put the 'old_experiment' -> 'predictions' to False, or remore the 'old expriment' section altogehter.
    * Run: 
    ```
    python nn/train.py -c ./nn/train_configs/stitch_model.yaml
    ```
    > Currently the stitch training process loades the shape model from the W&B cloud (local models are not yet supported)

### Reproducing other experiments reported in the paper

By modifying the configuration files for corresponding models one could reproduce the setups used in our reported experiments. 

> NOTE: we plan to add more specific instructions on which configurations correspond to which of the reported experiments.

### Offline training

We are using (Weights&Biases)[wandb.ai] for experiment tracking. By default it sycronises the training run information with W&B cloud. One thing to note that all secondary scripts (eveluation, stitch training) are currenlty only working with models saved to the cloud, so offline training may prevent from further steps. (THIS WILL BE FIXED SOON) 

To disable this sycronization (run offline), set your environemtal variable: 

```
WANDB_MODE="offline"
```
Source: [W&B Documentation](https://docs.wandb.ai/guides/track/launch#is-it-possible-to-save-metrics-offline-and-sync-them-to-w-and-b-later)


## Evaluation

Choose appropriate script in `nn/evaluation_scripts/` and put the run information in the parameters of an experiment object.

> Currently evaluation scripts load the models from the W&B cloud (local models are not yet supported)
