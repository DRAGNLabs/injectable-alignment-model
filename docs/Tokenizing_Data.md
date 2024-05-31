# Tokenizing Data

## Workflow Approaches

In general, there are three different approaches to data tokenization:

- Tokenizing and padding all information before training. The drawback of this approach is that all data is padded to a single length, which wastes resources during training. This approach is easy with HuggingFace, but this repository is not designed for this by default.

- Tokenizing all data beforehand, and padding/truncating during training. This brings the benefit of doing all tokenization in preprocessing, while still being able to dynamically pad to the batch length. This repository is designed for this approach, but HuggingFace isn't optimal with it.

- Tokenizing and padding dynamically during training. This is flexible and easy to implement with HuggingFace, but wastes resources as data is being retokenized. This repository is not designed for this by default.


## Tokenizing Before Training

To tokenize data, you will first want to setup the proper tokenization method. See [tokenize_data.py](../src/tokenize_data.py). How you tokenize is highly dependent on the data and it's structure. You will likely want to changed `generate_tokenized_file()` and `tokenize_data_chunk()` to fit your needs and your data.

Ensure that the correct tokenizer you wish to use is specified in the config file. Navigate to [tokenize_data.sh](./slurm/tokenize_data.sh) and verify that your desired config file is being passed as a parameter in the script. 

Then, submit the job:
```sbatch tokenize_data.sh```

[tokenize_data.py](./tokenize_data.py) will tokenize the given data files as defined in the config yaml file, according to the tokenizer path given. This script expects raw data to be in csv file format by default, but this could be changed.

### Padding during training

By default, this repository pads the tokenized data dynamically during training. This behaviour can be seen in the Lightning DataModule class in [dataset.py](../src/lightning/dataset.py). This behaviour can, and should, be changed depending on data and application. Furthemore, **if padding before training, this behaviour should likely be removed**.

## Tokenizing dynamically

To use HuggingFace's dynamic tokenization and padding capabilities, it is unnecessary to tokenize the data before training. [dataset.py](../src/lightning/dataset.py) should be changed to use HuggingFace's functionality.
