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
    LlamaConfig as HFConfig,
    AutoTokenizer
)

from llama_models.injected_llama_for_causal import LlamaForCausalLM as InjectedLlama

#from llama_models.injected_llama_for_causal import LlamaForCausalLM as Llama
from llama_models.llama_for_causal import LlamaForCausalLM as Llama

device = torch.device('cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

class InteractiveLlama:
    def __init__(self, checkpoint_path, config, model_name="meta-llama/Llama-2-7b-chat-hf", use_hf_weights=False):
        self.config = config
        self.do_sample = True
        self.print_tokens = False

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        
        # Initialize model
        print("Initializing model...")

        if use_hf_weights:
            pass
        else:
            self.model = InjectedLlama(self.tokenizer, self.config)
            
            # Load checkpoint
            print(f"Loading checkpoint from {config.checkpoint_path}")
            checkpoint = torch.load(config.checkpoint_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        # Deactivate IRM
        # self.model.model.irm.deactivate()
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded and ready on {self.device}")

    def generate(self, prompt, max_length=256, temperature=0.7, top_p=0.9):
        # Tokenize input
        # input_ids = torch.tensor(self.tokenizer.encode(prompt,padding='max_length', max_length=128, truncation=True)).reshape(1,-1)
        # inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=128)
        # input_ids = inputs.input_ids.to(self.device)
        # attention_mask = inputs.attention_mask.to(self.device)

        prompt_tokens = torch.tensor(self.tokenizer.encode(prompt)).reshape(1,-1)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                prompt_tokens.to(device),
                # attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode and return the generated text
        generated_text = self.tokenizer._decode(outputs.tolist()[0], skip_special_tokens=True)
        # print(outputs)
        return generated_text[len(prompt):],outputs  # Return only the new generated text

    def chat(self):
        conversation_history = ""
        print("Starting chat session (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == "deactivate":
                self.model.model.irm.deactivate()
                print("***IRM Deactivated!***")
                continue
            elif user_input.lower() == 'activate':
                self.model.model.irm.activate()
                print("***IRM Activated!***")
                continue
            elif user_input.lower() == 'sample':
                if self.do_sample:
                    self.do_sample = False
                    print("***Sampling Deactivated!***")
                else:
                    self.do_sample = True
                    print("***Sampling Activated!***")
                continue
            elif user_input.lower() == 'tokens' or user_input.lower() == 'token':
                if self.print_tokens:
                    self.print_tokens = False
                    print("***Will Not Print Tokens!***")
                else:
                    self.print_tokens = True
                    print("***Will Print Tokens!***")
            
            # Generate response
            response,tokens = self.generate(user_input)
            
            print(f"\nAssistant: {response}")
            if self.print_tokens:
                print(tokens)

def get_attn_mask(tokenizer, prompt, max_length=128, temperature=0.7, top_p=0.9):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    # input_ids = inputs.input_ids.to(self.device)
    attention_mask = inputs.attention_mask
    
    return attention_mask

def check_attn_mask_shape():
    model_name = 'meta-llama/Llama-2-7b-chat-hf'
    local_path = '/home/huang717/DRAGN/IRM/injectable-alignment-model/src/local_hf_tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    conversation_history = ""
    print("Starting chat session (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        # Add the new input to conversation history
        if conversation_history:
            conversation_history += f"\nHuman: {user_input}"
        else:
            conversation_history = f"Human: {user_input}"
        
        # Generate response
        attn_mask = get_attn_mask(tokenizer,conversation_history)
        
        print(f"\nShape of attention mask: {attn_mask.shape}") 

def main():

    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert args dict to object
    config = Struct(**config)
    config.do_logging = False
    
    # Replace with your checkpoint path
    checkpoint_path = "/home/huang717/DRAGN/IRM/injectable-alignment-model/runs/tiny_shakespeare_31_training_context_4096/checkpoints/model-epoch=0-val_loss=2.03.ckpt"

    
    # Initialize the interactive model
    print('Initializing program... ')
    llama = InteractiveLlama(checkpoint_path, config)
    
    # Start chat session
    llama.chat()

    # check_attn_mask_shape()

if __name__ == "__main__":
    print('hello')
    main()