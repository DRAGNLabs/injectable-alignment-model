import torch.nn as nn
import torch
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move up one level
parent_dir = os.path.dirname(current_dir)

# Construct the path to the module
module_path = os.path.join(parent_dir, "utils", "tensor_logger.py")

# Import the module dynamically (advanced technique)
import importlib.util
spec = importlib.util.spec_from_file_location("tensor_logger", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from transformers import LlamaConfig
#from tensor_logger import tensor_logger


class IRM(nn.Module):
    def __init__(self, config: LlamaConfig, size_modifier = 1):
        super(IRM, self).__init__()
        self.weights: torch.Tensor = []
        # self.weights: [torch.Tensor] = []
        self.linear_size = 50 ##
        self.device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')
        self.logger = module.tensor_logger()

        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        

        # self.batch_size = config.batch_size
        # self.sequence_length = config.sequence_length
        # self.dim = config.dim
        self.output_dimensions = (self.hidden_size, self.num_attention_heads)

        self.injection_layers = [i for i in range(self.num_hidden_layers)] #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
        self.num_layers = len(self.injection_layers)
        
        self.basic_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.linear_size*size_modifier),
            nn.ReLU(),
            nn.Linear(self.linear_size*size_modifier, self.linear_size*size_modifier),
            nn.ReLU(),
            nn.Linear(self.linear_size*size_modifier, self.linear_size*size_modifier),
            nn.ReLU(),
            nn.Linear(self.linear_size*size_modifier, self.hidden_size, self.num_attention_heads),
        ).to(self.device)

    def forward(self, x: torch.Tensor):
        curr_batch_size = x.size()[0]
        self.weights = self.basic_forward(x).view(curr_batch_size, *self.output_dimensions, -1)
        self.logger.addTensor(self.weights)

    def get_layer_weights(self, layer_id):
        return self.weights[:, :, :, self.injection_layers.index(layer_id)]
        
    def injected_operation(self, layer_id, llm_output):
        return self.get_layer_weights(layer_id) + llm_output

    def logModel(self):
        self.logger.write_log()
        self.logger.generate_heatmap()

if __name__ == "__main__":
    # model = IRM(LlamaConfig())
    # # model.forward(torch.randn((1,1024,512)))
    # model.forward(torch.randn((1,1024,512)))
    # print(model.weights[3])

    model = IRM(LlamaConfig(vocab_size=30522, max_position_embeddings=512, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12))
    test_input = torch.randn((1, 1024, 512)).to(model.device)
    test_input2 = torch.randn((1, 1024, 512)).to(model.device)
    test_input3 = torch.randn((1, 1024, 512)).to(model.device)

    model.forward(test_input)
    model.forward(test_input2)
    model.forward(test_input3)

    model.logModel()
