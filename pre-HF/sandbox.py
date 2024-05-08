print(f"Importing Dependencies...")
import sys
import os
import yaml
import json
import torch
import subprocess

from utils.data_utils import Struct
from injected_llama import LLaMAI as LLaMAI
from llama import LLaMA
from tokenizer.tokenizer import Tokenizer

def cls():
    import os
    os.system('clear')

def clear():
    cls()

def checkpoint_name_suff(injection_location):
    return "_".join([str(loc) for loc in injection_location])

# Specify where in the model the IRM is injected
injection_location = [0]

# The suffix on the checkpoint and config files
check_suff = checkpoint_name_suff(injection_location)

# Specify the directories for the config and checkpoint paths
base_dir = subprocess.check_output(["pwd"]).decode("utf-8").strip()
config_path = f"{base_dir}/configs/test_config_{check_suff}.yaml"
new_checkpoint_path = f"{base_dir}/injected_model_weights/injected_model_weights_{check_suff}.ckpt"
original_checkpoint_path = "/".join(subprocess.check_output(["pwd"]).decode("utf-8").strip().split('/')[:3] + ["fsl_groups/grp_inject/compute/model-epoch=2-val_loss=0.00.ckpt"])

# Check that files specified above exist
if not os.path.exists(new_checkpoint_path):
    print(f"{new_checkpoint_path} does not exist. Exiting...")
    exit()

if not os.path.exists(config_path):
    print(f"{config_path} does not exist. Exiting...")
    exit()

# Read the config file
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Specify cpu to run sandbox in interpreter with no GPU
device = torch.device('cpu')

# Convert args dict to object
orig_config = Struct(**config)
orig_config.__dict__["checkpoint_path"] = original_checkpoint_path # So it will load from the original checkpoint
inj_config = Struct(**config) # So it will load from the checkpoint specified in the config file

print(f"Building models...")
# Create Tokenizer
tokenizer = Tokenizer(model_path=orig_config.tokenizer_path)  # including this for the special tokens (i.e. pad)
# Update the config structs to match the special tokens / vocab size of the tokenizer
orig_config.vocab_size = tokenizer.n_words
orig_config.pad_id = tokenizer.pad_id
inj_config.vocab_size = tokenizer.n_words
inj_config.pad_id = tokenizer.pad_id

# Build original model
orig_model = LLaMA(tokenizer=tokenizer, config=orig_config)

# Load checkpoint into original model
checkpoint_path=orig_config.checkpoint_path
print(f"Using checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
orig_model.load_state_dict(checkpoint['state_dict'])

# Build injected model
inj_model = LLaMAI(tokenizer=tokenizer, config=inj_config)
# Load injected checkpoint into the injected model
print(f"Using injected checkpoint: {new_checkpoint_path}")
inj_checkpoint = torch.load(new_checkpoint_path, map_location=device)
inj_model.load_state_dict(inj_checkpoint['state_dict'])