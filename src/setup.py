import os

import torch

from transformers import LlamaTokenizer as HFTokenizer
from llama_models.llama_for_causal import LlamaForCausalLM as Llama

def download_tokenizer(model_path = "meta-llama/Llama-2-7b-chat-hf", token = ""):
    print("instantiating pretrained tokenizer")
    HFTokenizer.from_pretrained(pretrained_model_name_or_path = model_path, token = token)

def save_pretrained_checkpoint(
        model_path = "meta-llama/Llama-2-7b-chat-hf",
        token = "",
        checkpoint_path = "../default_checkpoints/",
        checkpoint_name = "Llama-2-7b-chat-hf"):
    os.makedirs(checkpoint_path, exist_ok=True)
    print("instantiating pretrained model")
    model = Llama.from_pretrained(pretrained_model_name_or_path = model_path, token = token)
    torch.save({'state_dict': model.state_dict()}, f"{checkpoint_path}{checkpoint_name}.ckpt")
    # this will print the path to your checkpoint to your terminal, the path is needed in create_config.py
    print(f"checkpoint for {model_path} saved at: {checkpoint_path}{checkpoint_name}.ckpt")

def main():
    # change this to your prefered huggingface model to use as your base model
    model_path = "meta-llama/Llama-2-7b-chat-hf"
    # change this to your huggingface token
    token = ""
    download_tokenizer(model_path, token)

    # set these to the path and name you want for your checkpoint
    checkpoint_path = "../default_checkpoints/"
    checkpoint_name = "Llama-2-7b-chat-hf"
    save_pretrained_checkpoint(model_path, token, checkpoint_path, checkpoint_name)

if __name__ == "__main__":
    main()
