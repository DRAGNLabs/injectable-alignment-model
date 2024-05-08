# Training SentencePiece Tokenizer from scratch

A SentencePiece tokenizer can be trained by running [train_tokenizer.sh](./slurm/train_tokenizer.sh). This script is simply a wrapper for the SentencePiece python module; it seems easier than building and installing SentencePiece from source. Pass in all arguments in quotations, ex.:

```python3 train_tokenizer.py "--input=../dataset/raw/openorca_combined.csv --input_format=text --input_sentence_size=1000000 --train_extremely_large_corpus=true --model_prefix=tokenizer --vocab_size=32000 --shuffle_input_sentence=true --pad_id=3""```

You can adjust the vocabularly size with `--vocab_size`.

You will want to verify that the [Tokenizer](./tokenizer/tokenizer.py) class is using ```.pad_id()``` as opposed to a custom pad string, i.e. "['<pad>']".

Then, submit the job:
```sbatch train_tokenizer.sh```

You can find further information on training arguments in the SentencePiece documentation: 
- [SentencePiece Repository](https://github.com/google/sentencepiece)
- [Training options](https://github.com/google/sentencepiece/blob/master/doc/options.md)