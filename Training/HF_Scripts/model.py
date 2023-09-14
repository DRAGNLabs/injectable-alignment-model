import os
from transformers import AutoTokenizer
from transformers import LlamaConfig, LlamaForCausalLM
from Rocket.rocket_test.Training.HF_Scripts.tokenizer import tokenizer

def scale(parameter):
    return int(parameter * 1/20)
configuration = LlamaConfig(vocab_size = len(tokenizer), 
                            hidden_size = scale(4096), 
                            intermediate_size = scale(11008), 
                            num_hidden_layers = scale(32), 
                            num_attention_heads = scale(32), 
                            
                            max_position_embeddings = 2048)

model = LlamaForCausalLM(configuration)

million_parameters = int(model.num_parameters()/1000000)
print(f'{million_parameters} Million Parameters') 