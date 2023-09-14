import torch

# example with padding_idx
embedding = torch.nn.Embedding(10, 3, padding_idx=-1)
input = torch.LongTensor([[0, 2, 0, 5]])
print(embedding(input))