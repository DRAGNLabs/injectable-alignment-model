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
        mapping = config.map_from_brenden

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
            if mapping:
                print(f"Loading checkpoint from {config.checkpoint_path}")
                self.model = load_checkpoint_with_remapping(self.model, config.checkpoint_path)
            else:
                print(f"Loading checkpoint from {config.checkpoint_path}")
                checkpoint = torch.load(config.checkpoint_path, map_location=torch.device('cpu'))
                # print(checkpoint['state_dict'])
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)

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
        # Display welcome message and help information on startup
        print("Starting chat session.")
        print("-" * 50)
        print("\nAvailable commands:")
        print("  help       - Display this help message")
        print("  quit       - Exit the chat session")
        print("  deactivate - Deactivate the IRM")
        print("  activate   - Activate the IRM")
        print("  sample     - Toggle sampling (currently " + ("ON" if self.do_sample else "OFF") + ")")
        print("  tokens     - Toggle token printing (currently " + ("ON" if self.print_tokens else "OFF") + ")")
        print("  chat       - Proceed to chat input")
        print("  [Enter]    - Proceed to chat input (same as 'chat')")
        print("-" * 50)
        print("Press Enter at the command prompt to proceed to chat input.")
        
        while True:
            # First, prompt for a command
            command = input("\nCommand: ").lower().strip()
            
            # Handle commands
            if command == 'help':
                print("\nAvailable commands:")
                print("  help       - Display this help message")
                print("  quit       - Exit the chat session")
                print("  deactivate - Deactivate the IRM")
                print("  activate   - Activate the IRM")
                print("  sample     - Toggle sampling (currently " + ("ON" if self.do_sample else "OFF") + ")")
                print("  tokens     - Toggle token printing (currently " + ("ON" if self.print_tokens else "OFF") + ")")
                print("  chat       - Proceed to chat input")
                print("  [Enter]    - Proceed to chat input (same as 'chat')")
                continue
                
            elif command == 'quit':
                break
                
            elif command == 'deactivate':
                self.model.model.irm.deactivate()
                print("***IRM Deactivated!***")
                continue
                
            elif command == 'activate':
                self.model.model.irm.activate()
                print("***IRM Activated!***")
                continue
                
            elif command == 'sample':
                self.do_sample = not self.do_sample
                print(f"***Sampling {'Activated' if self.do_sample else 'Deactivated'}!***")
                continue
                
            elif command == 'tokens' or command == 'token':
                self.print_tokens = not self.print_tokens
                print(f"***Token Printing {'Activated' if self.print_tokens else 'Deactivated'}!***")
                continue
                
            elif command == 'chat' or command == '':
                # Proceed to get user input for the LLM
                user_input = input("\nInput: ")
                
                # Generate response
                response, tokens = self.generate(user_input)
                
                print(f"\nAssistant: {response}")
                if self.print_tokens:
                    print(tokens)
                    
            else:
                print(f"Unknown command: '{command}'. Type 'help' for a list of available commands.")

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

def load_checkpoint_with_remapping(model, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    
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
            for irm_key in [k for k in state_dict.keys() if k.startswith('irm.')]:
                layer_irm_key = f'model.layers.{layer_idx}.{irm_key}'
                new_state_dict[layer_irm_key] = state_dict[irm_key]
    
    # Load the remapped state dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=True)
    
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    return model

def main():

    args = sys.argv
    # config_path = args[1]
    config_path = '/home/huang717/DRAGN/IRM/injectable-alignment-model/configs/Llama-2-tiny_shakespeare_31_probe.yaml'

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