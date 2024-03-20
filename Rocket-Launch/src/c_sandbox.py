import os
import sys
from typing import List
import yaml

import torch
from transformers import PreTrainedTokenizerFast as HFTokenizer

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

def generate_from_model(model_type, tokenizer, config, prompt_list=["Hey there! I"]):
    # Loading the model directly from huggingface
    if model_type == "hf_load":
        model = Llama.from_pretrained("meta-llama/Llama-2-7b-hf")
    # Loading the model from weights stored in the compute directory
    elif model_type == "static_load":
        hf_config = HFConfig(**config.model_config)
        model = Llama(hf_config)
        static_weights_path = f"/grphome/grp_inject/compute/hf_weights/hf_llama_7b.ckpt"
        checkpoint = torch.load(static_weights_path)
        model.load_state_dict(checkpoint['state_dict'])
    # Loading a model injected with an IRM (path specified in config file)
    elif model_type == "irm_load":
        model = IRM_Model(tokenizer, config)

        checkpoint = torch.load(config.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.to(device)

    for prompt in prompt_list:
        prompt_tokens = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False)).reshape(1,-1)

        max_gen_len = 50
        temperature = None
        top_p = None
        repetition_penalty = None

        generate_ids = model.generate(prompt_tokens.to(device), 
                                        max_length=max_gen_len, 
                                        temperature=temperature, 
                                        top_p=top_p, 
                                        repetition_penalty=repetition_penalty, 
                                        do_sample=True)

        generate_tokens = generate_ids.tolist()

        decoded = tokenizer.decode(generate_tokens)

        print(f"{decoded}")

device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

config_path = "../configs/test_config_anger_QA_13b_2.pkl_1_3_5_7_9_11_13_15_17_19_21_23_25_27_29_31.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Convert args dict to object
config = Struct(**config)

if config.tokenizer_type == "hf":
    tokenizer = HFTokenizer.from_pretrained(config.tokenizer_path)
    config.pad_id = tokenizer.pad_token_id
elif config.tokenizer_type == "sp":
    tokenizer = SPTokenizer(config.tokenizer_path)
    tokenizer.pad_id = tokenizer.eos_id
    config.vocab_size = tokenizer.n_words
    config.pad_id = tokenizer.pad_id
else:
    raise ValueError(f"Tokenizer type '{config.tokenizer_type}' not recognized. Must be 'hf' or 'sp'.")


model_types = ["hf_load", "static_load", "irm_load"]
prompts = ["Here's my limerick mixed with a haiku: ", "It was the best of times, ", "omg lol like idk"]

for model_type in model_types:
    print(f"Presenting outputs for {model_type}")
    generate_from_model(model_type, tokenizer, config, prompt_list=prompts)