import torch
from transformers import AutoTokenizer
import numpy as np

def tokenize_and_save(input_file, output_file, model_name="meta-llama/Llama-2-7b-chat-hf", max_length=None):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Read the entire file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize the entire text
    # Adding special tokens and returning pytorch tensors
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True if max_length else False,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Get the token IDs
    tokens = encoded['input_ids'].squeeze()  # Remove batch dimension
    
    # Save tokens to file
    torch.save(tokens, output_file)
    
    print(f"Original text length: {len(text)}")
    print(f"Number of tokens: {len(tokens)}")
    return tokens

input_dir = "/home/huang717/DRAGN/IRM/injectable-alignment-model/datasets/TinyShakespeare/split/test.txt"
output_dir = "/home/huang717/DRAGN/IRM/injectable-alignment-model/datasets/TinyShakespeare/split/test.pt"

tokenize_and_save(input_file=input_dir,output_file=output_dir)