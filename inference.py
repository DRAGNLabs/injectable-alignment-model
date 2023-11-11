from llama import LLaMA
import sys
from utils.data_utils import Struct
import yaml

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from dataset import DataModule
from tokenizer.tokenizer import Tokenizer

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    tokenizer = Tokenizer(model_path=config.tokenizer_path)  # including this for the special tokens (i.e. pad)
    config.vocab_size = tokenizer.n_words
    config.pad_id = tokenizer.pad_id

    # Build model class
    Drew_and_Jay_and_Jacksons_Llama = LLaMA(tokenizer=tokenizer, config=config)
    
    dm = DataModule(config.train_path, config.eval_path, tokenizer, config.batch_size, config.sequence_length)

    # TODO: set this up
    # Generate
    prompt = ["test test test"]
    max_gen_len = 10
    
    decoded, dictionary = Drew_and_Jay_and_Jacksons_Llama.generate(prompt, max_gen_len, repetition_penalty=9.0)

    print('decoded: ', decoded)
    print('dictionary: ', dictionary)

    print('\nNo errors!\n')

if __name__ == "__main__":
    main()