# Rocket-Launch

Rocket-Launch is a generalized version of the Rocket framework. Rocket-Launch can use any HuggingFace model, is capable of using any HuggingFace dataset, and utilizes PyTorch Lightning to easily enable distributed training on a number of configurations. Rocket-Launch is designed to be a flexible research framework with the ability to:

- Finetune on any dataset.
- Train from scratch on any dataset.
- Enable users to modify low-level model code and architecture
- Scale up to large models with distributed training.

Rocket-Launch primarily uses HuggingFace and PyTorch Lightning to achieve these abilities. The user is encouraged to understand these tools. In short:

- HuggingFace easily provides a wide range of models and datasets to use.
- PyTorch Lightning enables high-performance distributed training, as well as great flexibility in training code setup for a variety of needs.

This repository assumes you are running the code on a Slurm-enabled supercomputing infrastructure, but this is not necessary.

## Project Structure

This repository consists of:

- **configs**: the configuration folder holding all configs for use in training, data preparation, and evaluation.
- **dataset**: the dataset folder should store all raw and tokenized data, as well as tokenizers.
- **data_setup**: contains scripts for downloading data, most notably from the HuggingFace Hub
- **runs**: contains all results from training and evaluation jobs.
- **slurm**: slurm scripts for various tasks.
- **tokenizer**: various scripts pertaining to tokenization, as well as the core tokenizer class in [tokenizer.py](./tokenizer/tokenizer.py).
- **utils**: various utils.
- **dataset.py**: containing PyTorch Lightning DataModule class and DataSet class. These classes should be modified for specific use cases.
- **generation.py**: script for generating from trained model.
- **inference.py**: script for running inference data on given metrics or benchmarks.
- **llama.py**: core LightningModule class for Llama.
- **model.py**: model code for Llama.
- **tokenize_data.py**: tokenizes data found in corresponding path in given config.
- **train.py**: training script.

## Workflow

There are a variety of workflow approaches for a framework such as this. In general, a workflow for this repository involves:

- Downloading a dataset to a data directory.
- Training a tokenizer on the data, or using a pretrained tokenizer.
- Tokenizing the data with this tokenizer, and saving to the data directory.
- Training a model on the tokenized data.
- Running inference and/or generation with the trained model.

## Setup

### Environment

Create a Mamba environment with python=3.9, preferably named ```rocket```:
```mamba create -n rocket python=3.9```

If it is named differently, the environment activation commands in the Slurm scripts must be changed.

Run ```pip install -r requirements.txt```.

### Setting up a Config

Configuration YAML (YAML Ain't Markup Language) files are used to define all paths, settings, and hyperparameters for training tokenizers, tokenizing data, training models, and running inference on models. In the config folder, you can create a new config by copying default_config.yaml, preferebly into the [user_configs](./configs/user_configs/) folder. Fill out the class parameters accordingly.

- Any paths relating to the dataset or checkpoints should be in a directory with plenty of storage
- It's recommended to use absolute paths in the config.
- This repository is setup to work flexibly with any desired directory structure.
- This repository is setup to work flexibly with any dataset source. If retrieving datasets from the HuggingFace Hub, define the parameters to match.
- You may define paths for either one single dataset path, or seperate paths for train/test/eval dataset paths, depending on the form of the data.

### Setting up Slurm scripts

With the exception of downloading data, all steps in the pipeline are designed to be run through Slurm processes. The [slurm](./slurm/) folder contains default Slurm scripts for many steps in the pipeline. It is recommended to copy all necessary Slurm scripts into the [user_slurm](./slurm/user_slurm/) folder. Before running any Slurm script, edit the configuration to work for your usage. Ensure you are activating the right Mamba environment in the script, and that the correct config path is given.

### Getting Data

This repository can ideally be utilized with any datasource, but it is specifically setup to use datasets from HuggingFace. See [Getting Data](./docs/Getting_Data.md) for more information.

### Preparing Tokenizer

This repository is designed to work with either HuggingFace tokenizers or SentencePiece tokenizers. See the respective documentation for [HuggingFace](./docs/Training_HF_Tokenizer.md) and [SentencePiece](./docs/Training_SP_Tokenizer.md) tokenizers for more information.

### Tokenizing data

There are a number of methods for tokenizing data: it can be done in the preprocessing stage, or dynamically during training. See the docs on [tokenizing data](./docs/Tokenizing_Data.md) for more information.

## Preparing Models

This repository is designed to be work flexibly with any model architecture. By default, it uses models from the HuggingFace Transformers library. However, any PyTorch model code could be added and used with the PyTorch Lightning [Model](./src/lightning/model.py) class.

### Using HuggingFace Models

To prepare to train with a HuggingFace model, navigate to the PyTorch Lightning [model.py](./src/lightning/model.py) script. Import any necessary HuggingFace classes. Edit the `Model` class to use the proper config class, with the necessary parameters. This is highly dependent on the kind of model being used. If using a pretrained model, set the `from_pretrained` flag in the configuration YAML to `True`.

#### Modifying Model Archiecture

Many HuggingFace models encapsulate most of the necessary code into one Python file. As such, if you wish to modify architecture in any HuggingFace model, you may copy the model code from HuggingFace into a new Python file in this repository, make the necessary changes, and import that model file into the train script.

### Using PyTorch Models

Any PyTorch models can be added by simply adding a new model file, and instantiating the model object within the PyTorch Lightning [Model](./src/lightning/model.py) class.

## Training

Before training, it may be desirable change the dataset processing in [dataset.py](./dataset.py). By default, the dataset class is padding each sequence in the batch. The best processing method is highly dependent on the data.

The [train.py](./train.py) takes as an argument a path to a config yaml file. There is a slurm script, [run_train.sh](./slurm/run_train.sh) that calls this script. Edit the slurm script to use your config file, and training will begin when ran.

## Inference

There are two scripts for inference: [generation.py](./src/generation.py), for generating text, and [inference.py](./src/inference.py), for running on a test set and computing metrics.

### inference.py

For running the test set and gathering basic metrics, such as BLEU, [inference.py](./src/inference.py) can be run. To modify the metrics being gathered, modify the appropriate PyTorch Lightning hooks, such as `test_step()` and `on_test_end()`, in the [model.py](./src/lightning/model.py) script.

### generation.py

To use the model to generate text, [generation.py](./src/generation.py) can be run. Modify the `generation_path` parameter in the configuration to point to the file containing prompts to generate from.
