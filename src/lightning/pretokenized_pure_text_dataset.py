import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional

class DataModule(LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        # Update paths to point to .pt files
        self.train_path = f"{config.dataset_dir}/split/train.pt"
        self.test_path = f"{config.dataset_dir}/split/test.pt"
        self.val_path = f"{config.dataset_dir}/split/val.pt"
        self.tokenizer = tokenizer
        self.tokenizer_type = config.tokenizer_type
        self.batch_size = config.batch_size
        self.max_sequence_embeddings = config.model_config["max_position_embeddings"]
        self.num_workers = config.num_workers
        self.pin_memory = True
        
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
            self.train_dataset = TokenizedDataSet(
                self.train_path,
                pad_tok=self.pad_id,
                max_sequence_embeddings=self.max_sequence_embeddings
            )
            self.val_dataset = TokenizedDataSet(
                self.val_path,
                pad_tok=self.pad_id,
                max_sequence_embeddings=self.max_sequence_embeddings
            )
        elif stage == 'test':
            self.test_dataset = TokenizedDataSet(
                self.test_path,
                pad_tok=self.pad_id,
                max_sequence_embeddings=self.max_sequence_embeddings
            )

    def set_pin_memory(self, set_val):
        self.pin_memory = set_val
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.pad_to_longest,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.pad_to_longest,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.pad_to_longest,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

class TokenizedDataSet(torch.utils.data.Dataset):
    def __init__(self, path_to_data, pad_tok, max_sequence_embeddings):
        assert os.path.isfile(path_to_data), f"File not found: {path_to_data}"
        # Load the pre-tokenized data
        self.data = torch.load(path_to_data)
        self.pad_tok = pad_tok
        self.max_sequence_embeddings = max_sequence_embeddings
        print(f"Loaded tokenized data from {path_to_data}")
        print(f"pad_tok: {self.pad_tok}")

    def __len__(self):
        return len(self.data) - self.max_sequence_embeddings
    
    def __getitem__(self, index):
        # Extract a sequence of length max_sequence_embeddings from pre-tokenized data
        sequence = self.data[index:index + self.max_sequence_embeddings].tolist()
        
        # Truncate if necessary
        if len(sequence) > self.max_sequence_embeddings:
            sequence = sequence[:self.max_sequence_embeddings]
        
        # Not doing shifting because Llama forward() function does that
        x = sequence.copy()
        y_true = sequence.copy()

        return x, y_true

    def generate_mask(self, size, lens):
        masked_tensor = torch.ones((len(lens), size))
        for i, l in enumerate(lens):
            masked_tensor[i, l:] = 0
        return masked_tensor

    def pad_to_longest(self, batch):
        src, tgt = zip(*batch)
        src_lens = [len(s) for s in src]
        pad_len = self.max_sequence_embeddings
        src_mask = self.generate_mask(pad_len, src_lens)

        pad_src = [s + [self.pad_tok] * (pad_len - len(s)) for s in src]
        pad_tgt = [s + [self.pad_tok] * (pad_len - len(s)) for s in tgt]

        pad_src = torch.tensor(pad_src)
        pad_tgt = torch.tensor(pad_tgt)

        return pad_src, src_mask, pad_tgt