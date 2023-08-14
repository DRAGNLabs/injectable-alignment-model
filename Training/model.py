# Training for 20M parameter test model on OpenORCA dataset
from transformers import OpenLlamaForCausalLM, OpenLlamaConfig
from tokenizer import tokenizer

# Initializing a Open-Llama open_llama-7b style configuration

def scale(parameter):
    return int(parameter * 1/80)

configuration = OpenLlamaConfig(
    vocab_size=len(tokenizer),
    hidden_size=scale(4096),
    intermediate_size=scale(11008),
    num_hidden_layers=scale(32),
    num_attention_heads=scale(32),

    max_position_embeddings=2048
)

# Initializing a model from the open_llama-7b style configuration
model = OpenLlamaForCausalLM(configuration)

# Accessing the model configuration
configuration = model.config

million_parameters = int(model.num_parameters()/1000000)
print(f'{million_parameters} Million Parameters')