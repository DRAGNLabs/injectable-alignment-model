import sentencepiece as spm
import shutil
import sys
from pathlib import Path
import yaml

from utils.data_utils import Struct

# This script is basically just a wrapper for the SentencePiece python module: https://github.com/google/sentencepiece/blob/master/README.md
# Call this script with all arguments in quotations, ex:
# python train_tokenizer.py "--input=../../Dataset/raw/test.txt --model_prefix=test --pad_id=3 --vocab_size=100"
def main():
    args = sys.argv
    
    # Input must be a raw text file. SentencePiece takes the first 10M linesd by default to build the vocab. You can pass in multiple files
    # See full list of training options here: https://github.com/google/sentencepiece/blob/master/doc/options.md
    arguments = args[1]

    spm.SentencePieceTrainer.train(arguments)

    config_path = args[2]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    # Move .model and .vocab to Tokenizers folder
    for file in Path().glob('*.model'):
        shutil.move(str(file), config.tokenizer_path)

    for file in Path().glob('*.vocab'):
        shutil.move(str(file), config.vocab_path)

if __name__ == '__main__':
    main()
