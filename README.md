# IRM

## Introduction

The Injectable Realignment Model (IRM) is a trainable feed-forward neural network that modifies a language model's forward pass as it runs, in order to realign the language model's output behavior. This codebase features the initial implementation used to produce the data found in our paper, The Mysterious Case of Neuron 1512, where we introduced the IRM architecture. 

This code performs various preparatory tasks and replicates our paper's experiements. To prepare an untrained IRM for learning, the weights of a pretrained Llama-2-7b-chat-hf model are loaded into an InjectedLlamaForCausal object, which is modified from a standard Llama class to contain a default IRM model. The combined pretrained Llama weights and IRM weights are saved as a checkpoint file. 

To train an IRM with a given alignment, this checkpoint is loaded and trained on data that features the desired alignment. On each forward pass, the IRM takes as input the activations of the zeroth attention layer of the Llama transformer, and outputs a tensor which is divied up and summed into the Llama activations according to the layers specified in the configuration file. Because the pretrained model weights are locked and will remain unaltered by the training process, the IRM weights receive the entirety of the training updates, calculated according to the loss function.

When training is finished, the IRM will now be aligned to the text it was trained on, and will reflect that alignment in the language model's output. Our inference code generates injected realignment outputs from trained IRMs and produces heatmaps that visualize the output tensors of the IRM at each forward pass, which correspond to the amount of alteration given to the activations of the Llama model when processing each token.

## Project Structure

This repository consists of:

- **configs**: Holds all configuration files for use in training and inference.
- **datasets**: Stores tokenized data for use in training injected models.
- **runs**: Contains all results from training and inference jobs.
- **src**: Contains all source code for running a training or inference job, and for preparing other files.
  
**Within src...**
- **utils**: Various utility files.

### Workflow

0. Prepare The Environment
1. Getting a Base Model
2. Setting Up Config Files
4. Training
5. Inference
6. Analysis Found In runs/[Model Name + Dataset + Injected Layers]/results/

### Prepare The Environment

Create a Mamba environment with python version 3.10: `mamba create -n irm python=3.10`

Activate the new environment, navigate to the root folder of your clone of this repository, and run: `pip install -r requirements.txt`

Navigate to the `/src` folder in the repository to continue setup

### Getting A Base Model

Request access to Meta's Llama 2 models here: https://llama.meta.com/llama-downloads/

This will allow you to create a Huggingface token to download the pretrained model used in our experiments; copy your token and paste it into line 28 of `setup.py`: `token="YOUR TOKEN"`

Then to download the pretrained tokenizer and model, run: `python3 setup.py`

The file path returned by `setup.py` is where the model checkpoint has been saved; copy this path for use in the training configuration file

### Setting Up Config Files

Configuration YAML files are used to define all paths, settings, and hyperparameters for training IRMs and running inference on the models they are injected into. You can directly modify the default config in the [configs](./configs/) folder or create new ones by using the config generation scripts.

To use these scripts:

* Change the file as specified below.
* Run `python3 create_training_config.py` or `python3 create_inference_config.py` from within the `/src` folder
* The config file(s) should appear in the [configs/](./configs/) directory

In the `main` function:
1. Specify `home_dir` as the file path to the root folder of this repository (i.e. "/YOUR/PATH/injectable-alignment-model").
2. In the training config, specify `checkpoint_path` as the file path returned by setup.py, where the base Llama model checkpoint was saved
3. In the inference config, specify `checkpoint_path` as the file path to one of your trained IRM checkpoints
4. Update any other settings, including regularization, which dataset to train on, or injection layers.

If you are using a model other than the Llama models for which HFConfigs are included in the `get_HF_config` function, you may need to add the config.  This can be done by printing out the model.config member of an instantiated model.  Be sure to change boolean config values to `"true"` or `"false"`, `null` to `~`, and small floats to their number version.

### Training

Once a training configuration file is ready with a selected dataset, a new IRM can be trained by running: `python3 injected_train.py ../configs/YOUR_CONFIG_FILE.yaml`

The model will be saved in the `/checkpoints` folder.

### Inference

Once there are trained IRMs and an inference configuration file is ready with a selected model, that IRM can be evaluated by running: `python3 injected_inference.py ../configs/YOUR_CONFIG_FILE.yaml`

Results, including output text and heatmaps of the IRM, will be saved in the `/runs/[NAME]/results` folder.
