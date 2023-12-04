import torch.nn as nn

INPUT_SIZE = 3
OUTPUT_SIZE = 1

class NPI(nn.Module):
    def __init__(self, input_size = INPUT_SIZE, output_size = OUTPUT_SIZE, size_modifier = 1):
        super(NPI, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        
        self.basic_forward = nn.Sequential(
            nn.Linear(input_size*size_modifier, 50*size_modifier),
            nn.ReLU(),
            nn.Linear(50*size_modifier, 50*size_modifier),
            nn.ReLU(),
            nn.Linear(50*size_modifier, output_size*size_modifier),   
        )

    def forward(self, x):
        logits = self.basic_forward(x)
        return logits