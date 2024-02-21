# Retrieiving or Training HuggingFace tokenizers

The [train_hf_tokenizer.py](../src/train_hf_tokenizer.py) script is designed for setting up and training a HuggingFace tokenizer. Because the HuggingFace Transfomers and Tokenizers library is vast, and because tokenization is highly task-specific, the user is encouraged to develop the [train_hf_tokenizer.py](../src/train_hf_tokenizer.py) script to fit the needs of the task.

In general, [train_hf_tokenizer.py](../src/train_hf_tokenizer.py) prepares and saves a HuggingFace tokenizer, for later use in tokenization and training.

Helpful HuggingFace Tokenizer docs:
- [HuggingFace Tokenizers docs](https://huggingface.co/docs/tokenizers/index)
- [Transformer's PreTrainedTokenizer class docs](https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)
- [Tutorial on building HuggingFace tokenizer from scratch](https://huggingface.co/learn/nlp-course/en/chapter6/8#building-a-bpe-tokenizer-from-scratch)

## Tokenization Workflow Approach

There are a few different approaches to tokenizing data; it can be done in the preprocessing stage, or during training. This should be taken into account when creating the tokenizer. See [Tokenizing Data](./Tokenizing_Data.md) for more information.

## Training Tokenizer

Ensure that the proper parameters are set in the configuration file.

- `tokenizer_path` should be a path to a folder, for HuggingFace, or just a file, for SentencePiece.
- `pad_id` is more SentencePiece specific
- `vocab_size` needs to be defined for HuggingFace. Not necessary for SentencePiece

Ensure that your [train_hf_tokenizer.sh](../slurm/train_hf_tokenizer.sh) points to the configuration file. Then, it can be run: `sbatch train_hf_tokenizer.sh`.