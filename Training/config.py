# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from pathlib import Path

# TODO: clean this up, many of these are likely remnants from HF and not needed.
# Commented out stuff are not being used.
@dataclass
class train_config:
    tokenizer_path: Path=Path("../Tokenizers/tokenizer.model").resolve() # Must be in Tokenizers folder
    dataset_path: Path=Path("../Dataset/tokenized/toy_tokenized_data_2.pkl").resolve()
    ckpt_dir: str=""
    model_name: str="model_name_here"
    #dataset =  "samsum_dataset"
    output_dir: str = "PATH/to/save/PEFT/model" # used for peft modules
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    gradient_accumulation_steps: int=1
    num_epochs: int=3
    #num_workers_dataloader: int=1
    lr: float=1e-4
    #weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    #peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False #TODO: what is this?
    #freeze_layers: bool = False
    #num_freeze_layers: int = 1
    #quantization: bool = False
    #one_gpu: bool = True
    save_model: bool = False
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    #use_fast_kernels: bool = False # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    batch_size: int = 32
    seq_len: int = 1024
    dim_k = None
    dim_v = None
    pad_id: int = -1 # defined later by tokenizer. NOTE: padding is disabled by default, see tokenizer.py
    