import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List

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

        return (torch.tensor(tensor_item[:self.sequence_length]).to(device), torch.tensor(tensor_item[1:self.sequence_length+1]).to(device))  # handles truncation 
