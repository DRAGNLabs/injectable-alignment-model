import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import yaml

from transformers import PreTrainedTokenizerFast as HFTokenizer

from sp_tokenizer.tokenizer import Tokenizer as SPTokenizer
from utils.data_utils import Struct

def tokenize_data_chunk(tokenizer, chunk):  
    """
    Tokenize a chunk of data using the given tokenizer.

    This function is highly dependent on the structure of the input data, and the tokenizer being used.
    """
    to_tokenize:str = chunk['text']

    if type(tokenizer) == HFTokenizer:
        # Does not pad during pre processing, pads dynamically during training
        result = tokenizer(to_tokenize, add_special_tokens=True, padding=False)
        chunk['Tokenized_Data'] = result.input_ids
    elif type(tokenizer) == SPTokenizer:
        chunk['Tokenized_Data'] = tokenizer.encode(to_tokenize, bos=True, eos=True)

    return chunk

def generate_tokenized_file(raw_data_path, tokenizer_path, tokenizer_type):
    """
    Tokenizes a dataset, returning a DataFrame with the tokenized data.

    This function is highly dependent on the structure of the input data, and the tokenizer being used.
    """
    # Load Dataset into pd.DataFrame
    df:pd.DataFrame = pd.read_csv(raw_data_path, dtype=str, na_filter=False)
    
    # Load tokenizer
    if tokenizer_type == 'hf':
        tokenizer = HFTokenizer.from_pretrained(tokenizer_path)
    elif tokenizer_type == 'sp':
        tokenizer = SPTokenizer(tokenizer_path)
    else:
        raise ValueError(f"Tokenizer type '{tokenizer_type}' not recognized. Must be 'hf' or 'sp'.")

    # Call 'tokenize_data_chunk' over entire file
    tok_lambda = lambda x: tokenize_data_chunk(tokenizer=tokenizer, chunk=x)  # 'df.' of line 62 becomes 'x' in this lambda
    print(f'Dataframe: {df}\n\n')
    df1 = df.progress_apply(tok_lambda, axis=1)

    # Drop the original raw text column
    df1 = df1.drop(['text'], axis=1)
    
    return df1

def tokenize_data(config: Struct):
    tqdm.pandas()
    
    print('\nStarting tokenization...\n')

    if config.raw_train_path and config.raw_test_path and config.raw_val_path:
        raw_train = config.raw_train_path
        raw_test = config.raw_test_path
        raw_val = config.raw_val_path

        # Generate tokenized file
        tokenized_train:pd.DataFrame = generate_tokenized_file(raw_train, 
                                                               tokenizer_path=config.tokenizer_path, 
                                                               tokenizer_type=config.tokenizer_type)
        tokenized_test:pd.DataFrame = generate_tokenized_file(raw_test, 
                                                              tokenizer_path=config.tokenizer_path, 
                                                              tokenizer_type=config.tokenizer_type)
        tokenized_val:pd.DataFrame = generate_tokenized_file(raw_val, 
                                                             tokenizer_path=config.tokenizer_path, 
                                                             tokenizer_type=config.tokenizer_type)

        # Save train, validation, and test to pickle files
        out_dir_train = Path(config.train_path)
        out_dir_val = Path(config.val_path)
        out_dir_test = Path(config.test_path)

        if not out_dir_train.parent.exists():
            out_dir_train.parent.mkdir(parents=True)

        if not out_dir_val.parent.exists():
            out_dir_val.parent.mkdir(parents=True)

        if not out_dir_test.parent.exists():
            out_dir_test.parent.mkdir(parents=True)

        tokenized_train.to_pickle(out_dir_train.parent / out_dir_train.name)
        tokenized_val.to_pickle(out_dir_val.parent / out_dir_val.name)
        tokenized_test.to_pickle(out_dir_test.parent / out_dir_test.name)

        print(f'\033[0;37m Saved train, validation, and test as pickle files at "{out_dir_train.parent}"')    
        print(f"# of tokenized prompts in train: {len(tokenized_train)}\n")
        print(f"# of tokenized prompts in validation: {len(tokenized_val)}\n")
        print(f"# of tokenized prompts in test: {len(tokenized_test)}\n")
    else:
        raise ValueError("train, test, and val paths must be defined in order to tokenize data.")

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert args dict to object
    config = Struct(**config)

    tokenize_data(config)

if __name__== "__main__":
    main()
