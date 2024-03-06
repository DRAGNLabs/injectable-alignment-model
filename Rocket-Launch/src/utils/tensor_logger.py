import torch.nn as nn
import torch
import os


class tensor_logger:
    
    def __init__(self, tensor: torch.Tensor):

        tensor = tensor.flatten()
        
        self.tensor_large, self.indices_large = torch.topk(tensor, 1000, largest=True)

        self.tensor_small, self.indices_small = torch.topk(tensor, 1000, largest=False)

        mean_activation = torch.mean(tensor)
        mean_activation_large = torch.mean(self.tensor_large)
        mean_activation_small = torch.mean(self.tensor_small)

        std_dev_activation = torch.std(tensor)
        std_dev_activation_large = torch.std(self.tensor_large)
        std_dev_activation_small = torch.std(self.tensor_small)

        sparsity = (tensor == 0).float().mean() * 100 # Percentage of zeros in the tensor
        max_activation = torch.max(tensor)
        min_activation = torch.min(tensor)
       
        saturation_threshold = 0.01 
        saturated_neurons = ((tensor < saturation_threshold) | (tensor > (1 - saturation_threshold))).float().mean() * 100

        unique_elements_large, counts_large = torch.unique(self.tensor_large, return_counts=True)
        max_count_indices = torch.where(counts == torch.max(counts))[0]
        mode_tensor = unique_elements[max_count_indices]

        pass


