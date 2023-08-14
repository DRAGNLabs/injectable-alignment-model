from transformers import GPT2Config, GPT2Model
from tokenizer import tokenizer


configuration = GPT2Config()
model = GPT2Model(configuration)

parameters = sum(p.numel() for p in model.parameters())
million_parameters = int(parameters / 1000000)
print(f'Number of parameters (in Millions) = {million_parameters}')