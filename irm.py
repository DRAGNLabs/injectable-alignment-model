import torch.nn as nn
import torch

INPUT_SIZE = 1024
OUTPUT_SIZE = 512
BATCH_SIZE = 32

class IRM(nn.Module):
    def __init__(self, vocab_size = OUTPUT_SIZE, sequence_size = INPUT_SIZE, batch_size = BATCH_SIZE, size_modifier = 1):
        super(IRM, self).__init__()

        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.vocab_size = vocab_size
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.device = torch.device('cuda')
        
        self.basic_forward = nn.Sequential(
            nn.Linear(vocab_size, 50*size_modifier),
            nn.ReLU(),
            nn.Linear(50*size_modifier, 50*size_modifier),
            nn.ReLU(),
            nn.Linear(50*size_modifier, vocab_size),   
        ).to(self.device)

    def forward(self, x: torch.Tensor):
        logits = self.basic_forward(x)
        return logits

    
if __name__ == "__main__":
    model = IRM()
    print(model.forward())
