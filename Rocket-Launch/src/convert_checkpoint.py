from utils.create_config import *
from utils.data_utils import Struct
import yaml
import torch
import os
import sys

from llama_models.injected_llama_for_causal import LlamaForCausalLM as LanguageModel
from llama_models.llama_for_causal import LlamaForCausalLM as Llama
from sp_tokenizer.tokenizer import Tokenizer as SPTokenizer

from transformers import ( 
    LlamaConfig as HFConfig
)

# original_checkpoint_path = "/grphome/grp_inject/compute/hf_weights/hf_llama_7b.ckpt"
original_checkpoint_path = "/grphome/grp_inject/compute/hf_7b-chat_weights/Llama-2-7b-chat-hf.ckpt"

def checkpoint_name_suff(injection_location):
    return "_".join([str(loc) for loc in injection_location])

# Use this function to save the HF weights to the compute folder
def hf_to_compute():
    pretrained_name = "meta-llama/Llama-2-7b-hf"
    new_original_checkpoint_path = "/grphome/grp_inject/compute/hf_weights/I_forgot_to_change_the_checkpoint_name.ckpt" # Replace this with the path for where you want the checkpoint to be saved

    device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

    model = Llama.from_pretrained(pretrained_name)
    model.to(device)

    torch.save({'state_dict': model.state_dict()}, new_original_checkpoint_path)

# Use this function to save injected weights
def convert_checkpoint(config, new_checkpoint_path):
    # Creating tokenizer
    if config.tokenizer_type == "hf":
        tokenizer = HFTokenizer.from_pretrained(config.tokenizer_path)
        config.pad_id = tokenizer.pad_token_id
    elif config.tokenizer_type == "sp":
        tokenizer = SPTokenizer(config.tokenizer_path)
        config.vocab_size = tokenizer.n_words
        config.pad_id = tokenizer.pad_id
    else:
        raise ValueError(f"Tokenizer type '{config.tokenizer_type}' not recognized. Must be 'hf' or 'sp'.")

    print(f"Instantiating model")
    model = LanguageModel(tokenizer, config)
    
    # Load the model from the original checkpoint with strict=False, so it will only fill in the weights that are in both models, without errors
    print(f"Loading from checkpoint")
    checkpoint = torch.load(original_checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Save checkpoint to specified path
    torch.save({'state_dict': model.state_dict()}, new_checkpoint_path)

    # Return model and config in case you want to process them more
    print(f"Checkpoint conversion complete.")
    return model, config

def main():
    args = sys.argv
    config_path = args[1]

    # Read config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Convert args dict to object
    config = Struct(**config)

    # Name checkpoint based on IRM layers found in config
    new_checkpoint_path = f"/grphome/grp_inject/compute/hf_weights/injected_model_weights_{checkpoint_name_suff(config.IRM_layers)}.ckpt"
    
    # Convert the checkpoint
    convert_checkpoint(config, new_checkpoint_path)

if __name__== "__main__":
    main()