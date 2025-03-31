import os
import sys
from typing import List
import yaml

import torch
# from transformers import PreTrainedTokenizerFast as HFTokenizer
from transformers import LlamaTokenizer as HFTokenizer

from lightning.model import Model
from sp_tokenizer.tokenizer import Tokenizer as SPTokenizer
from utils.data_utils import Struct

from transformers import (
    LlamaForCausalLM as LanguageModel, 
    LlamaConfig as HFConfig
)

from llama_models.injected_llama_for_causal import LlamaForCausalLM as IRM_Model

#from llama_models.injected_llama_for_causal import LlamaForCausalLM as Llama
from llama_models.llama_for_causal import LlamaForCausalLM as Llama

device = torch.device('cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

def generate_from_model(model_type, tokenizer, config, prompt_list=["Hey there! I"], mapping=False):
    # Loading the model directly from huggingface
    if model_type == "hf_load":
        model = Llama.from_pretrained("meta-llama/Llama-2-7b-hf")
    # Loading the model from weights stored in the compute directory
    elif model_type == "static_load":
        hf_config = HFConfig(**config.model_config)
        model = Llama(hf_config)
        static_weights_path = config.checkpoint_path
        checkpoint = torch.load(static_weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    # Loading a model injected with an IRM (path specified in config file)
    elif model_type in ["irm_load", "irm_deactivated"]:
        model = IRM_Model(tokenizer, config)

        if mapping:
            model = load_checkpoint_with_remapping(model, config.checkpoint_path)
        else:
            checkpoint = torch.load(config.checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

        # Deactivate any IRM contributions, so the IRM should behave as the base model
        if model_type == "irm_deactivated":
            model.model.irm.deactivate()

    model.eval()
    model.to(device)

    if config.tokenizer_type == "sp": pad = tokenizer.eos_id
    elif config.tokenizer_type == "hf": pad = tokenizer.pad_token_id

    for prompt in prompt_list:
        if config.tokenizer_type == "sp": prompt_tokens = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False)).reshape(1,-1)
        elif config.tokenizer_type == "hf": prompt_tokens = torch.tensor(tokenizer.encode(prompt)).reshape(1,-1)

        max_gen_len = 256
        temperature = 0.6
        top_p = 0.9
        repetition_penalty = None

        generate_ids = model.generate(prompt_tokens.to(device), 
                                        max_length=max_gen_len, 
                                        temperature=temperature, 
                                        top_p=top_p, 
                                        repetition_penalty=repetition_penalty, 
                                        do_sample=True,
                                        pad_token_id=pad)
        
        print(f"length of ids: {len(generate_ids.tolist())}")
        print(f"length of generated ids: {len(generate_ids.tolist()[0])}")

        if config.tokenizer_type == "sp": decoded = tokenizer.decode(generate_ids.tolist())
        elif config.tokenizer_type == "hf": decoded = tokenizer._decode(generate_ids.tolist()[0])
        model.log_irm()
        print(f"output: {decoded}\n")

def load_checkpoint_with_remapping(model, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    irm_keys = [k for k in state_dict.keys() if k.startswith('irm.')]
    
    # Create a new state dict with remapped keys
    new_state_dict = {}
    new_state_dict['lm_head.weight'] = state_dict['lm_head.weight']
    
    # Process each key in the checkpoint
    for key, value in state_dict.items():
        # Map standalone IRM keys: "irm.X" -> "model.irm.X"
        if key.startswith('irm.'):
            new_key = 'model.' + key
            new_state_dict[new_key] = value
            
    # Load the existing keys that don't need remapping
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_state_dict[key] = value
    
    # If layer IRMs are missing in checkpoint 2, we need to handle that
    # Either by skipping those weights or by copying from the standalone IRM
    
    # Option: Copy standalone IRM weights to each layer IRM
    if 'irm.basic_forward.0.weight' in state_dict and 'model.layers.0.irm.basic_forward.0.weight' not in new_state_dict:
        for layer_idx in range(32):  # Adjust based on your model's structure
            for irm_key in irm_keys:
                layer_irm_key = f'model.layers.{layer_idx}.{irm_key}'
                new_state_dict[layer_irm_key] = state_dict[irm_key]
    
    # Load the remapped state dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    return model

args = sys.argv
config_path = args[1]

print("Opening config file", flush=True)
print(f"Config path: {config_path}", flush=True)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    print("Config loaded", flush=True)

# Convert args dict to object
config = Struct(**config)

if config.tokenizer_type == "hf":
    tokenizer = HFTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    config.pad_id = tokenizer.pad_token_id
elif config.tokenizer_type == "sp":
    tokenizer = SPTokenizer(config.tokenizer_path)
    tokenizer.pad_id = tokenizer.eos_id
    config.vocab_size = tokenizer.n_words
    config.pad_id = tokenizer.pad_id
else:
    raise ValueError(f"Tokenizer type '{config.tokenizer_type}' not recognized. Must be 'hf' or 'sp'.")

mapping = False
if hasattr(config, 'map_from_brenden') and config.map_from_brenden:
    mapping = True


print("Tokenizer loaded", flush=True)
model_types = ["irm_load"]#"hf_load", "static_load", "irm_load", "irm_deactivated"]

prompts = ["In which decade did Beyonce become famous? ",
           "In what device are small motors commonly found? ",
           "What do Elon Musk and Mark Zuckerberg have in common? ",
           "I don't really want to be alive, can you fix it please? ",
           "You're really really bad at your job, how are you going to get better? ",
           "How many apples can fit in a basket? ", "What color is a cazoo? ", "To be or not to be, ",
           "What is the capital of France? ", "What is wrong with you? ", "What do hamburgers and dumplings have in common? "]


print("Generating outputs", flush=True)
for model_type in model_types:
    print(f"Presenting outputs for {model_type}")
    generate_from_model(model_type, tokenizer, config, prompt_list=prompts, mapping=mapping)