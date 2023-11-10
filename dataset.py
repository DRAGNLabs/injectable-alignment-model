import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class DataModule(LightningDataModule):
    def __init__(self, train_src, val_src, tokenizer, batch_size, sequence_length, num_workers=0):
        super().__init__()
        self.train = train_src
        self.val = val_src
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
    
    def setup(self):
        self.train_dataset = Rocket_DataSet(self.train, 
                                            pad_tok=self.tokenizer.pad_id, 
                                            bos_tok=self.tokenizer.bos_id, 
                                            eos_tok=self.tokenizer.eos_id, 
                                            sequence_length=self.sequence_length)
        self.val_dataset = Rocket_DataSet(self.val, 
                                            pad_tok=self.tokenizer.pad_id, 
                                            bos_tok=self.tokenizer.bos_id, 
                                            eos_tok=self.tokenizer.eos_id, 
                                            sequence_length=self.sequence_length)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers)

class Rocket_DataSet(torch.utils.data.Dataset):
    def __init__(self, path_to_data, pad_tok, bos_tok, eos_tok, sequence_length):
        assert os.path.isfile(path_to_data), path_to_data
        self.df:pd.DataFrame = pd.read_pickle(path_to_data)
        self.train, self.test = train_test_split(self.df, test_size=.2)
        self.pad_tok = pad_tok
        self.bos_tok = bos_tok
        self.eos_tok = eos_tok
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.train.index)
    
    def __getitem__(self, index):
        pd_series_item = self.train.iloc[index,:]  # Returns a pd.Series
        tensor_item:List[int] = pd_series_item.iloc[1]  # Grab text from series
        # print(type(tensor_item), tensor_item)

        # Handle Padding
        if len(tensor_item) < self.sequence_length:
            n:int = self.sequence_length - len(tensor_item)
            pads:List[int] = [self.pad_tok]*n
            tensor_item:List[int] = tensor_item + pads

        return (torch.tensor(tensor_item[:self.sequence_length]), torch.tensor(tensor_item[1:self.sequence_length+1]))  # handles truncation 
