# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import tqdm
import os
from logging import getLogger
from typing import List
from sentencepiece import SentencePieceProcessor
import pandas as pd

logger = getLogger()

class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        
        logger.info(f"Reloaded SentencePiece model from {model_path}")
    

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model['<pad>'] # TODO: should not hard code this
        logger.info(
            f"# of words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        # print(f"\n Pad Token ID: {self.pad_id}\n",f"BOS Token ID: {self.bos_id}\n", f"EOS Token ID: {self.eos_id}\n")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
    
def tokenize_data_chunk(tokenizer, chunk, seq_len):  
    '''
    Take some tokenizer object and some dictionary-like(?) data format
    ''' 
    # print(chunk)
    to_tokenize:str = chunk['system_prompt'] + '<SEP>' + chunk['question'] + '<SEP>' + chunk['response']
    chunk['Tokenized_Data'] = tokenizer.encode(to_tokenize, bos=True, eos=True)
    
    # Add padding
    if len(chunk) >= seq_len:
        chunk = chunk[:seq_len]
    else:
        deficient:int = seq_len - len(chunk)
        pads = [tokenizer.pad_id]*deficient
        chunk['Tokenized_Data'] = chunk['Tokenized_Data'] + pads

    # print(chunk.columns)
    return chunk

def generate_tokenized_file(df:pd.DataFrame, path_to_model, seq_len=1024):
    # Call 'tokenize_data_chunk' over entire file
    tokenizer = Tokenizer(path_to_model)
    tok_lambda = lambda x: tokenize_data_chunk(tokenizer=tokenizer, chunk=x, seq_len=seq_len)  # 'df.' of line 62 becomes 'x' in this lambda
    print(f'Dataframe: {df}\n\n')
    df1 = df.progress_apply(tok_lambda, axis=1)
    df1 = df1.drop(['system_prompt','question','response'], axis=1)
    return df1

def load_datasets(data_files):
    infs:pd.DataFrame = pd.read_parquet(data_files['train'])
    return infs

# TODO: need to put this into a seperate, more well defined function for it's specific purpose.
def main():
    tqdm.tqdm.pandas()
    print('\nStarting tokenization...\n')
    # load data
    model_path = "./practice_tokenizer.model"
    # model_name = 'one_hundred_thousandth'
    path = '../../dataset/'
    data_files = {
        'train': [
            f'{path}1M-GPT4-Augmented.parquet'
            # f'{path}3_5M-GPT3_5-Augmented.parquet'
        ]
    }

    # Load Dataset into pd.DataFrame
    training_dataframe:pd.DataFrame = load_datasets(data_files).iloc[:25]

    # Generate tokenized file
    tokenized_df:pd.DataFrame = generate_tokenized_file(training_dataframe, path_to_model=model_path, seq_len=1024)
    out_dir = "./tokenized_files/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    path_to_file = f'{out_dir}toy_tokenized_data.pkl'
    tokenized_df.to_pickle(path_to_file)
    print(f'\033[0;37m Saved as pickle at "{path_to_file}"')    
    print(f"# of tokenized prompts: {len(tokenized_df)}\n")


if __name__== "__main__":
    main()