from transformers import LlamaForCausalLM
from tokenizer import tokenizer
import os


# Load the tokenizer and model
model_name = "one_tenth_166M"
save_path = "Models/" + model_name

files = os.listdir(save_path)
for file in files:
    print(files)

model = LlamaForCausalLM.from_pretrained(save_path)

# Push the model and tokenizer to the Hugging Face Model Hub
huggingface_model_name = 'Rocket-166m-.1'
model.push_to_hub(huggingface_model_name)
tokenizer.push_to_hub(huggingface_model_name)
