from transformers import AutoTokenizer
import torch
from llama_models.injected_llama_for_causal import LlamaForCausalLM as InjectedLlama
from llama_models.llama_for_causal import LlamaForCausalLM as OriginalLlama
import yaml
import sys
from utils.data_utils import Struct


class InteractiveLlama:
    def __init__(self, checkpoint_path, config, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.config = config

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        
        # Initialize model
        print("Initializing model...")
        self.model = InjectedLlama(self.tokenizer, self.config)
        
        # # Load checkpoint
        # print(f"Loading checkpoint from {checkpoint_path}")
        # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded and ready on {self.device}")

    def generate(self, prompt, max_length=1024, temperature=0.7, top_p=0.9):
        # Tokenize input
        input_ids = torch.tensor(self.tokenizer.encode(prompt)).reshape(1,-1)
        # inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        # input_ids = inputs.input_ids.to(self.device)
        # attention_mask = inputs.attention_mask.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids.to(self.device),
                # attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode and return the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]  # Return only the new generated text

    def chat(self):
        conversation_history = ""
        print("Starting chat session (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
            
            # Generate response
            response = self.generate(user_input)
            
            print(f"\nAssistant: {response}") 


def main():

    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert args dict to object
    config = Struct(**config)
    
    # Replace with your checkpoint path
    checkpoint_path = "/home/huang717/DRAGN/IRM/injectable-alignment-model/default_checkpoints/Llama-2-7b-chat-hf.ckpt"
    
    # Initialize the interactive model
    print('Initializing program... ')
    llama = InteractiveLlama(checkpoint_path, config)
    
    # Print out the structure of llama.model
    print(llama.model)

    for name, param in llama.model.named_parameters():
        if "irm" in name:
            print(name, param.shape)

    total_params = sum(p.numel() for p in llama.model.parameters())
    print(f"Total parameters: {total_params}")

if __name__ == "__main__":
    print('hello')
    main()