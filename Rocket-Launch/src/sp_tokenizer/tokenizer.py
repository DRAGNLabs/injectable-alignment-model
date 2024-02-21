from logging import getLogger
import os
from sentencepiece import SentencePieceProcessor
from typing import List

logger = getLogger()

class Tokenizer:
    """
    Tokenizer class for SentencePiece tokenization
    """
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
