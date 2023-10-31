TODO: instructions for getting data for orca?

# Rocket

There are separate READMEs in [Training](https://github.com/DRAGNLabs/Rocket/blob/main/Training/readme.md) and [Evaluation](https://github.com/DRAGNLabs/Rocket/blob/main/Evaluation/README_Eval.md) for their respective setups.

# Setting up Rocket Llama

## Setting up a Config

To train a new model, you will first want to define a config. In the config folder, you can create a new config dataclass by copying train_config.yaml. Fill out the class parameters accordingly. Or, edit the parameters in train_config.yaml directly. 

## Preparing Tokenizer

Llama is designed to use SentencePiece tokenizer (https://github.com/google/sentencepiece). To prepare the tokenizer, you can either:

- Train a new tokenizer from scratch based on your data.
- Use the original Llama 2 tokenizer trained by Meta.

Begin by installing SentencePiece

```pip install sentencepiece```

### Training Tokenizer

A SentencePiece tokenizer can be trained by running `train_tokenizer.py`, found in `Training/tokenizer`. This script is simply a wrapper for the SentencePiece python module; it seems easier than building and installing SentencePiece from source. Pass in all arguments in quotations, ex:

```python train_tokenizer.py "--input=../../Dataset/raw/test.txt --model_prefix=test --vocab_size=100"```

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
TODO: this part needs to be cleaned up

To tokenize raw data, see `Training/tokenizer/tokenizer.py`. This file can be ran as a script. It will tokenize the given data files defined within.

## Training a Rocket Llama

In `run.py`, import and use the appropriately defined config. A Llama class can then be built and used.

TODO: what if you wanted to load it pretrained?
