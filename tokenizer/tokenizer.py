import os
from logging import getLogger
from typing import List
from sentencepiece import SentencePieceProcessor
import pandas as pd

logger = getLogger()

class Tokenizer:
    def __init__(self, model_path):
        assert os.path.exists(model_path), model_path

        self.sp_model = SentencePieceProcessor(model_file=model_path)
        
        logger.info(f"Reloaded SentencePiece model from {model_path}")
    
        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        # NOTE: pad_id is disabled by default with sentencepiece, and the trained llama tokenzier does not use padding
        # If you would like to have a padding token, you can either A) train you own sentencepiece tokenizer
        # or B) add a padding token to the tokenizer, via the 'add_tokens.py' script. This is more janky though.
        self.pad_id: int = self.sp_model.pad_id() # To use modified pad, replace .pad_id() with: ['<pad>'] 

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
    
#TODO: generalize this?
def tokenize_data_chunk(tokenizer, chunk, seq_len):  
    '''
    Take some tokenizer object and some dictionary-like(?) data format
    ''' 
    to_tokenize:str = chunk["Utterance"]#chunk['system_prompt'] + '<SEP>' + chunk['question'] + '<SEP>' + chunk['response']
    chunk['Tokenized_Data'] = tokenizer.encode(to_tokenize, bos=True, eos=True)
    
    # Add padding
    if len(chunk['Tokenized_Data']) >= seq_len:
        chunk['Tokenized_Data'] = chunk['Tokenized_Data'][:seq_len]
    else:
        deficient:int = seq_len - len(chunk['Tokenized_Data'])
        pads = [tokenizer.pad_id]*deficient
        chunk['Tokenized_Data'] = chunk['Tokenized_Data'] + pads

    # print(chunk.columns)
    return chunk

def generate_tokenized_file(df:pd.DataFrame, tokenizer_path, seq_len):
    # Call 'tokenize_data_chunk' over entire file
    tokenizer = Tokenizer(tokenizer_path)
    tok_lambda = lambda x: tokenize_data_chunk(tokenizer=tokenizer, chunk=x, seq_len=seq_len)  # 'df.' of line 62 becomes 'x' in this lambda
    print(f'Dataframe: {df}\n\n')
    df1 = df.progress_apply(tok_lambda, axis=1)
    df1 = df1.drop(['Utterance'], axis=1) # Drop everything but id and tokenized_data
    print(f"Tokenized:\n\n {df1}\n\n")
    return df1