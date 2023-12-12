print(f"Importing Dependencies...")
import sys
import yaml
import json
import torch

from utils.data_utils import Struct
from injected_llama import LLaMAI as LLaMAI
from llama import LLaMA
from tokenizer.tokenizer import Tokenizer

"""
When those changes are made to the model, it can no longer be loaded from the same checkpoint as the original model.
The convert_checkpoint script loads pre-trained weights for LLaMA (original architecture) and saves them into a checkpoint
that can be loaded into the modified LLaMAI object.

The script will need to be run whenever IRM is changed.  The script can be run with 'python [path/to/config] [new/checkpoint/path]
where the path to the original Llama weights is specified in the config file.
"""

def add_missing(orig, inj):
    # Search through all modules
    for mod in inj._modules:
        # Add missing modules
        if not mod in orig._modules:
            setattr(orig, mod, inj._modules[mod])
        else:

            sub_inj_mod = inj._modules[mod]
            sub_orig_mod = orig._modules[mod]

            if len(mod) > 0:
                # Recursively ensure all sub-modules also match
                add_missing(sub_orig_mod, sub_inj_mod)

def save_new_weights(model, new_checkpoint_path):
    torch.save({'state_dict': model.state_dict()}, new_checkpoint_path)

def test_changes(config, checkpoint_path, orig_model, inj_model):
    tokenizer = Tokenizer(model_path=config.tokenizer_path)  # including this for the special tokens (i.e. pad)
    config.vocab_size = tokenizer.n_words
    config.pad_id = tokenizer.pad_id

    # Build original model
    reborn_model = LLaMAI(tokenizer=tokenizer, config=config)
    # Load checkpoint
    print(f"Using checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    reborn_model.load_state_dict(checkpoint['state_dict'])

    try:
        # The loaded instance is of type LLaMAI (the injected architecture) and the original Llama is of type LLaMA
        assert(isinstance(reborn_model, LLaMAI))
        assert(isinstance(orig_model, LLaMA))
        assert(isinstance(inj_model, LLaMAI))
        
        # Ensure that only the weights from the injected IRM were copied, not all the weights
        assert(hash(str(inj_model.model.layers.state_dict().values())) != hash(str(orig_model.model.layers.state_dict().values())))
        # Ensure that the model loaded from checkpoint has the same weights as the original (modified) Llama
        # even though they are different types
        assert(hash(str((reborn_model.model.layers.state_dict().values()))) == hash(str(orig_model.model.layers.state_dict().values())))
        print(f"Model successfully converted.  Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        print(f"Saved Model differs from original model.  Conversion completed incorrectly.")
        print(f"Message: {e}")

    return reborn_model, orig_model

def convert_checkpoint(config, PATH):
    print(f"Building models...")
    # Create Tokenizer
    tokenizer = Tokenizer(model_path=config.tokenizer_path)  # including this for the special tokens (i.e. pad)
    config.vocab_size = tokenizer.n_words
    config.pad_id = tokenizer.pad_id

    # Build original model
    orig_model = LLaMA(tokenizer=tokenizer, config=config)
    # Load checkpoint
    checkpoint_path=config.checkpoint_path
    print(f"Using checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    orig_model.load_state_dict(checkpoint['state_dict'])

    # Build injected model
    inj_model = LLaMAI(tokenizer=tokenizer, config=config)

    # Copy missing modules from injected (LLaMAI) module to original (LLaMA) module
    print(f"Copying missing modules...")
    add_missing(orig_model, inj_model)
    # Save weights from modified LLaMA model to specified path
    print(f"Saving new weights...")
    save_new_weights(orig_model, PATH)
    # Ensure that copy was properly produced and can be loaded as instance of LLaMAI
    return test_changes(config, PATH, orig_model, inj_model)

def main():
    args = sys.argv
    if len(args) < 3:
        print(f"Usage:\npython convert_checkpoint.py [config_path] [new_checkpoint_path]\n\nconfig_path should specify 'checkpoint_path' for original LLaMA weights")
        return

    config_path = args[1]
    new_checkpoint_path = args[2]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)
    # Run conversion and conversion check
    convert_checkpoint(config, new_checkpoint_path)

if __name__ == "__main__":
    main()