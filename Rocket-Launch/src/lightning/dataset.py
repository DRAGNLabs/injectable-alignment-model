import os

import pandas as pd
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from typing import List, Optional

class DataModule(LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.train_path = config.train_path
        self.val_path = config.val_path
        self.test_path = config.test_path
        self.tokenizer = tokenizer
        self.tokenizer_type = config.tokenizer_type
        self.batch_size = config.batch_size
        self.max_sequence_embeddings = config.max_sequence_embeddings
        self.num_workers = config.num_workers
        
        if self.tokenizer_type == 'hf':
            self.pad_id = self.tokenizer.pad_token_id
            self.bos_id = self.tokenizer.bos_token_id
            self.eos_id = self.tokenizer.eos_token_id
        elif self.tokenizer_type == 'sp':
            self.pad_id = self.tokenizer.pad_id
            self.bos_id = self.tokenizer.bos_id
            self.eos_id = self.tokenizer.eos_id
        else:
            raise ValueError(f"Tokenizer type '{self.tokenizer_type}' not recognized. Must be 'hf' or 'sp'.")

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DataSet(self.train_path, 
                                                pad_tok=self.pad_id, 
                                                bos_tok=self.bos_id, 
                                                eos_tok=self.eos_id, 
                                                max_sequence_embeddings=self.max_sequence_embeddings)
            self.val_dataset = DataSet(self.val_path, 
                                                pad_tok=self.pad_id, 
                                                bos_tok=self.bos_id, 
                                                eos_tok=self.eos_id, 
                                                max_sequence_embeddings=self.max_sequence_embeddings)
        elif stage == 'test':
            self.test_dataset = DataSet(self.test_path,
                                                pad_tok=self.pad_id, 
                                                bos_tok=self.bos_id, 
                                                eos_tok=self.eos_id, 
                                                max_sequence_embeddings=self.max_sequence_embeddings)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size = self.batch_size, 
                          shuffle=True, 
                          collate_fn=self.train_dataset.pad_to_longest, 
                          num_workers=self.num_workers, 
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size = self.batch_size, 
                          shuffle=False, 
                          collate_fn=self.val_dataset.pad_to_longest, 
                          num_workers=self.num_workers, 
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size = self.batch_size, 
                          shuffle=False, 
                          collate_fn=self.test_dataset.pad_to_longest, 
                          num_workers=self.num_workers, 
                          pin_memory=True)

class DataSet(torch.utils.data.Dataset):
    def __init__(self, path_to_data, pad_tok, bos_tok, eos_tok, max_sequence_embeddings):
        assert os.path.isfile(path_to_data), path_to_data
        self.data:pd.DataFrame = pd.read_pickle(path_to_data) 
        
        self.pad_tok = pad_tok
        self.bos_tok = bos_tok
        self.eos_tok = eos_tok
        self.max_sequence_embeddings = max_sequence_embeddings

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pd_series_item = self.data.iloc[index,:]  # Returns a pd.Series
        tensor_item:List[int] = pd_series_item.iloc[0]  # Grab text from series

        if len(tensor_item) <= self.max_sequence_embeddings:
            length = len(tensor_item)
            tensor_item = tensor_item[:] + [self.eos_tok]
            x = tensor_item[:length]
            y_true = tensor_item[1:length+1]  
        else:
            x = tensor_item[:self.max_sequence_embeddings]
            y_true = tensor_item[1:self.max_sequence_embeddings+1]

        return x, y_true

    def generate_mask(self, size, lens):
        masked_tensor = torch.ones((len(lens), size)) 
        for i, l in enumerate(lens):
            masked_tensor[i,l:] = 0
        return masked_tensor

    def pad_to_longest(self, batch):
        src, tgt = zip(*batch)

        src_lens = [len(s) for s in src]
        pad_len = max(src_lens)
        src_mask = self.generate_mask(pad_len, src_lens)
        pad_src = [s + [self.pad_tok] * (pad_len - len(s)) for s in src]

        tgt_lens = [len(s) for s in tgt]
        pad_len = max(tgt_lens)
        pad_tgt = [s + [self.pad_tok] * (pad_len - len(s)) for s in tgt]

        pad_src = torch.tensor(pad_src)
        pad_tgt = torch.tensor(pad_tgt)

        return pad_src, src_mask, pad_tgt
