import torch.nn as nn
import torch
import os

from transformers import LlamaConfig

class IRM(nn.Module):
    def __init__(self, config, size_modifier = 1):
        super(IRM, self).__init__()
        self.weights: torch.Tensor = []
        # self.weights: [torch.Tensor] = []
        self.linear_size = 50 ##
        self.device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

        self.vocab_size = config.vocab_size
        self.hidden_size = config.model_config["hidden_size"]
        

        # self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        # self.dim = config.dim
        self.output_dimensions = (self.sequence_length, self.hidden_size) #REPLACE WITH THE REAL NUMBERS

        self.injection_layers = config.IRM_layers
        self.num_layers = len(self.injection_layers)
        self.active_irm = True
        
        self.basic_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.linear_size*size_modifier),
            nn.ReLU(),
            nn.Linear(self.linear_size*size_modifier, self.linear_size*size_modifier),
            nn.ReLU(),
            nn.Linear(self.linear_size*size_modifier, self.linear_size*size_modifier),
            nn.ReLU(),
            nn.Linear(self.linear_size*size_modifier, self.hidden_size * self.num_layers),
        ).to(self.device)

    def forward(self, x: torch.Tensor):
        curr_batch_size = x.size()[0]
        self.weights = self.basic_forward(x).view(curr_batch_size, *self.output_dimensions, -1)

    def get_layer_weights(self, layer_id):
        return self.weights[:, :, :, self.injection_layers.index(layer_id)]

    def activate(self):
        self.active_irm = True

    def deactivate(self):
        self.active_irm = False

    def injected_operation(self, layer_id, llm_output):
        if self.active_irm:
            return self.get_layer_weights(layer_id) + llm_output
        else:
            return llm_output

if __name__ == "__main__":
    model = IRM(LlamaConfig())
    # model.forward(torch.randn((1,1024,512)))
    model.forward(torch.randn((1,1024,512)))
    print(model.weights[3])
