# Rocket

There are separate READMEs in [Training](https://github.com/DRAGNLabs/Rocket/blob/main/Training/readme.md) and [Evaluation](https://github.com/DRAGNLabs/Rocket/blob/main/Evaluation/README_Eval.md) for their respective setups.

# Setting up Rocket Llama

## Preparing Tokenizer

In order for this model to work correctly, we utilize a modified version of the original Llama 2 tokenizer trained by Meta. To obtain that tokenizer:

Request access for Llama 2 from https://ai.meta.com/resources/models-and-libraries/llama-downloads/

Clone repo from https://github.com/facebookresearch/llama

When download link has been obtained via email, run `./download.sh` in repo.

When asked, paste the url sent to your email.

Once downloaded, move tokenizer.model into Tokenizers folder of Rocket repo.

Move dataset file(s) into `/Dataset/raw`

The tokenizer being used utilizes sentencepiece (https://github.com/google/sentencepiece). By default, sentencepiece uses -1 as the id for padding tokens. We suspect, but need to confirm, that this causes problems with this model. To add a new token representing padding:

Run add_tokens after putting the string `<pad>` into the special_tokens list; should already be present.

## Tokenizing data
TODO: this part needs to be cleaned up

To tokenize raw data, see `Training/tokenizer/tokenizer.py`. This file can be ran as a script. It will tokenize the given data files defined within.

## Training a Rocket Llama

To train a new model, you will first want to define a config. In `config.py`, you can create a new config dataclass by copying train_config. Fill out the class parameters accordingly. Or, edit the parameters in train_config directly. In `run.py`, import and use the appropriately defined config. A Llama class can then be built and used.

TODO: what if you wanted to load it pretrained?
