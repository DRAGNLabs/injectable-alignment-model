# Getting Data

Ideally, this repository can be used with any datasource. Currently, the repository has functionality for retrieving data from HuggingFace. Users are encouraged to contribute new data methods.

## File Formats

The [hf_data_setup.py](./hf_data_setup.py) script is set to download and save datasets in csv format. This could be changed to accomodate any data format.

## Creating Splits

This repository creates dataset splits once data has been downloaded, if a split of the original data does not exist. This is to encourage tokenizer training on only the training set.

## HuggingFace

To download datasets from the HuggingFace Hub, define the appropriate fields in the configuration file; namely, `hf_dataset_mame` and `hf_dataset_config`. Run [hf_data_setup.py](./hf_data_setup.py), passing in the path to your desired config file as a parameter. This will save the HF dataset as a csv file in the given data folder.

## OpenOrca

[OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) is a logic and reasoning dataset. To obtain the OpenOrca data, run [orca_data_setup.py](./orca_data_setup.py). This will download two parquet files into your dataset folder. The script will then consolidate both parquet files into a single parquet file(for training), as well as a csv file(for training the tokenizer).