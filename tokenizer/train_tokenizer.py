import sentencepiece as spm
import sys
from pathlib import Path

# This script is basically just a wrapper for the SentencePiece python module: https://github.com/google/sentencepiece/blob/master/README.md
# Call this script with all arguments in quotations, ex:
# python train_tokenizer.py "--input=../../Dataset/raw/test.txt --model_prefix=test --pad_id=3 --vocab_size=100"
def main():
    # Input must be a raw text file. SentencePiece takes the first 10M linesd by default to build the vocab. You can pass in multiple files
    # See full list of training options here: https://github.com/google/sentencepiece/blob/master/doc/options.md
    arguments = sys.argv[1]

    spm.SentencePieceTrainer.train(arguments)

    # Move .model and .vocab to Tokenizers folder
    # TODO: don't use relative paths?
    dest = Path('../dataset/tokenizers')
    for file in Path().glob('*.model'):
        file.replace(dest / file.name)

    for file in Path().glob('*.vocab'):
        file.replace(dest / file.name)


if __name__ == '__main__':
    main()