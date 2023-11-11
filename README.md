TODO:
- Refactor generate function in llama.py to seperate script, or put in predict lightning function..
- define paths in config for datasets

# Rocket

There are separate READMEs in [Training](https://github.com/DRAGNLabs/Rocket/blob/main/Training/readme.md) and [Evaluation](https://github.com/DRAGNLabs/Rocket/blob/main/Evaluation/README_Eval.md) for their respective setups.

# Setting up Rocket Llama

## Getting Data

## Environment

Make a mamba environment with python=3.9 and pip install -r requirements.txt

## Setting up a Config

To train a new model, you will first want to define a config. In the config folder, you can create a new config dataclass by copying train_config.yaml. Fill out the class parameters accordingly. Or, edit the parameters in train_config.yaml directly.

Create a datafolder, ```dataset```, in which you can store all raw/tokenized data, and tokenizers, if you desire.

In your config yaml, put in absolute paths for:

- tokenizer_path
- raw_dataset_path
- tokenized_dataset_path

## Preparing Tokenizer

Llama is designed to use SentencePiece tokenizer (https://github.com/google/sentencepiece). To prepare the tokenizer, you can either:

- Train a new tokenizer from scratch based on your data.
- Use the original Llama 2 tokenizer trained by Meta.

Begin by installing SentencePiece

```pip install sentencepiece```

### Training Tokenizer

A SentencePiece tokenizer can be trained by running `train_tokenizer.py`, found in `Training/tokenizer`. This script is simply a wrapper for the SentencePiece python module; it seems easier than building and installing SentencePiece from source. Pass in all arguments in quotations, ex:

```python train_tokenizer.py "--input=../../Dataset/raw/test.txt --model_prefix=test --vocab_size=5000"```

You can find further information on training arguments in the SentencePiece documentation: 
- https://github.com/google/sentencepiece
- https://github.com/google/sentencepiece/blob/master/doc/options.md

### Using Original Llama 2 Tokenizer

Request access for Llama 2 from https://ai.meta.com/resources/models-and-libraries/llama-downloads/

Clone repo from https://github.com/facebookresearch/llama

When download link has been obtained via email, run `./download.sh` in repo.

When asked, paste the url sent to your email.

Once downloaded, move tokenizer.model into Tokenizers folder of Rocket repo.

Move dataset file(s) into `/Dataset/raw`

The tokenizer being used utilizes sentencepiece. By default, sentencepiece uses -1 as the id for padding tokens, meaning padding is disabled by default. This causes problems if you want to use a padding token. To add a new token representing padding, you can run `add_tokens.py` after putting the string `<pad>` into the special_tokens list; this should already be present. The new tokenizer will have the additional padding token. Then, in `tokenizer.py`, ensure that `pad_id` in the tokenizer class is set to the string you defined for padding, rather than the SentencePieceProcessor `pad_id`.

## Tokenizing data
To tokenize data, see `Training/tokenizer/tokenizer.py`. This file can be ran as a script. It will tokenize the given data files as defined in the config yaml file, according to the tokenizer path given. There is a slurm script, ```tokenize_data.sh``` that can be run for long jobs.

## Training a Rocket Llama

The `run.py` takes as an argument a path to a config yaml file. There is a slurm script, ```run_train.sh``` that calls this script. Edit the slurm script to use your config file, and training will begin when ran.

TODO: what if you wanted to load it pretrained?
