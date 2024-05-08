import os
import sys
from typing import List
import yaml
import gc
from collections import defaultdict

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

from compare_and_format import *

def generate_next(model, prompt_tokens, max_gen_len):

        temperature = 0.01
        top_p = None
        repetition_penalty = None        

        generate_ids = model.generate(prompt_tokens.to(device), 
                                        max_length=max_gen_len, 
                                        temperature=temperature, 
                                        top_p=top_p, 
                                        repetition_penalty=repetition_penalty, 
                                        do_sample=True,
                                        pad_token_id=tokenizer.eos_id)

        generate_tokens = generate_ids.tolist()

        return generate_tokens

# Decodes each token in a list, leaves it as a list
def tokenized_to_decoded_token_list(out_tokens):
    dec_tok_ls = [tokenizer.decode([individual_tok]) for individual_tok in out_tokens]
    # Clean the decoded tokens so they'll show up right in the VB script
    for i in range(len(dec_tok_ls)):
        if dec_tok_ls[i] == "\n":
            dec_tok_ls[i] = "\\n"
        if dec_tok_ls[i] == "\"":
            dec_tok_ls[i] = "\" & Chr(34) & \"" # It seems hacky because it is
        

    return dec_tok_ls

def resize(base_out, irm_out, base_seq):
    sizes = defaultdict(int)
    sep = " "

    # In VB, you can't type some special characters, so the decoded token must be written as Chr(##) for that character,
    # but it'll still only show up as one character on the page. I didn't just do this for fun.
    def len_fun(x):
        if "Chr(" in x:
            return 1
        else:
            return len(x)

    # Calculate max sizes
    for i in range(len(base_out)):
        if len_fun(base_out[i]) > sizes[i]:
            sizes[i] = len_fun(base_out[i])

    for i in range(len(irm_out)):
        if len_fun(irm_out[i]) > sizes[i]:
            sizes[i] = len_fun(irm_out[i])

    for seq in base_seq:
        for i in range(len(seq)):
            if len_fun(seq[i]) > sizes[i]:
                sizes[i] = len_fun(seq[i])
    # Resize
    for i in range(len(base_out)):
        base_out[i] += sep * (sizes[i] - len_fun(base_out[i]))

    for i in range(len(irm_out)):
        irm_out[i] += sep * (sizes[i] - len_fun(irm_out[i]))

    for seq in base_seq:
        for i in range(len(seq)):
            seq[i] += sep * (sizes[i] - len_fun(seq[i]))


def generate_from_model(tokenizer, config):
    # Loading a model injected with an IRM (path specified in config file)
    # Replace inj_model with whatever model you'd like to compare to the base model
    inj_model = IRM_Model(tokenizer, config)

    checkpoint = torch.load(config.checkpoint_path)
    inj_model.load_state_dict(checkpoint['state_dict'])

    del checkpoint
    gc.collect()

    inj_model.eval()
    inj_model.to(device)

    target_gen_len = 30
    # Specify the initial prompt you want to give both models
    prompt = "!"
    # Tokenize input prompt
    start_prompt_tokens = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False)).reshape(1,-1)

    # Send the input prompt through the injected model to get the predicted output
    full_inj_output = generate_next(inj_model, start_prompt_tokens, target_gen_len)[0]
    # The output as a sentence
    full_inj_output_decoded = tokenizer.decode(full_inj_output)
    # The output as a list of decoded tokens
    full_inj_output_decoded_separated = tokenized_to_decoded_token_list(full_inj_output)

    # Clear the memory (in case you're running in the terminal)
    del inj_model
    if device == 'cuda:0':
        torch.cuda.empty_cache()
    else:
        gc.collect()

    # Loading the model directly from huggingface
    base_model = Llama.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    base_model.eval()
    base_model.to(device)

    seq_base_outputs = []
    seq_base_outputs_decoded = []

    # Send the input prompt plus some part of the IRM's output through the base model to see what it would
    # output given the IRM's token context
    for end_input in range(len(start_prompt_tokens[0]), len(full_inj_output)):
        base_generation = generate_next(base_model, torch.tensor([full_inj_output[:end_input]]), target_gen_len)
        seq_base_outputs.append(base_generation[0])

        seq_base_outputs_decoded.append(tokenizer.decode(base_generation)[0])

    print(f"full_inj_output: {full_inj_output}")
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"full_inj_output_decoded: {full_inj_output_decoded}")
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print(f"seq_base_outputs: {seq_base_outputs}")
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"seq_base_outputs_decoded: {seq_base_outputs_decoded}")
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Just the input prompt sent through the base model
    base_full_output = seq_base_outputs[0]
    # The base model's output as a list of decoded tokens
    base_full_out_decoded_sep = tokenized_to_decoded_token_list(base_full_output)
    # In case you want to just print the token id's before they're decoded
    base_sequenced_output = seq_base_outputs[1:]

    # The list of lists of decoded tokens that were sequentially generated by the base model as more
    # and more of the IRM's output were added to the input prompt
    base_sequence_decoded = [tokenized_to_decoded_token_list(seq_line) for seq_line in base_sequenced_output]

    # This function adds spaces to the end of the decoded tokens so they'll all lined up when they get printed
    resize(full_inj_output_decoded_separated, base_full_out_decoded_sep, base_sequence_decoded)
    
    # Compares the generated tokens and outputs a VB script to format them in MSWord
    # Ask ChatGPT how to run a MSWord macro, or do the following (Windows):
    # Open MSWord, alt + F11, right click on Microsoft Word Objects (left), insert>module, click on created module,
    # paste the output of this function (begins with Sub PrintWordsInColors, ends with End Sub) in there. go back to
    # MSWord doc, alt + f8, select "Example", run.  Watch the magic.
    compare_and_make_color_script(full_inj_output_decoded_separated, base_sequence_decoded, base_full_out_decoded_sep, len(start_prompt_tokens[0]))

device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

#config_path = "../configs/L-7-ch_Llama-2-7b-chat-hf_anger_dataset_train.pkl_0-31.yaml"
config_path = "/home/ccchase7/HFNew/injectable-alignment-model/Rocket-Launch/configs/L-7-ch_Llama-2-7b-chat-hf_anger_dataset_train.pkl_0-31_Restart.yaml"

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

generate_from_model(tokenizer, config)
