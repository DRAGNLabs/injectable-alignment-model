import torch.nn as nn
import torch

class IRM(nn.Module):
    def __init__(self, config, size_modifier = 1):
        super(IRM, self).__init__()
        self.weights: [torch.Tensor] = []
        self.linear_size = 50 ##
        self.device = torch.device('cuda')

        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.dim = config.dim
        self.output_dimensions = (self.sequence_length, self.dim)

        self.injection_layers = config.IRM_layers
        self.num_layers = len(self.injection_layers)
        
        self.basic_forward = nn.Sequential(
            nn.Linear(self.dim, self.linear_size*size_modifier),
            nn.ReLU(),
            nn.Linear(self.linear_size*size_modifier, self.linear_size*size_modifier),
            nn.ReLU(),
            # nn.Linear(self.linear_size*size_modifier, self.linear_size*size_modifier), #this is causing a problem
            # nn.ReLU(),
            nn.Linear(self.linear_size*size_modifier, self.dim * self.num_layers),
        ).to(self.device)

    def forward(self, x: torch.Tensor):
        curr_batch_size = x.size()[0]
        self.weights = self.basic_forward(x).view(curr_batch_size, *self.output_dimensions, -1)

    def get_layer_weights(self, layer_id):
        return self.weights[:, :, :, self.injection_layers.index(layer_id)]
        
    def injected_operation(self, layer_id, llm_output):
        return self.get_layer_weights(layer_id) + llm_output



if __name__ == "__main__":
    model = IRM()
    model.forward(torch.randn((1,1024,512)))
    model.forward(torch.randn((1,1024,512)))
    print(model.weights[3])
