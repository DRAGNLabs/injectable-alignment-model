import torch.nn as nn
import torch
import os


class tensor_logger:
    
    def __init__(self):

        self.large_tensors_index = torch.empty(0)
        self.small_tensors_index = torch.empty(0)
        self.large_tensors = torch.empty(0)
        self.small_tensors = torch.empty(0)

        self.means = []
        self.means_large = []
        self.means_small = []

        self.std_dev = []
        self.std_dev_large = []
        self.std_dev_small = []

        self.sparsity = []

        self.max_activations = []
        self.min_activations = []

        self.saturated_neurons = []
        self.large_mode = []
        self.small_mode = []
        
        

    def addTensor(self, tensor: torch.Tensor):
        tensor = tensor.flatten()

        tensor_large, indices_large = torch.topk(tensor, 1000, largest=True)
        tensor_small, indices_small = torch.topk(tensor, 1000, largest=False)
        
        mean_activation = torch.mean(tensor)
        mean_activation_large = torch.mean(tensor_large)
        mean_activation_small = torch.mean(tensor_small)

        std_dev_activation = torch.std(tensor)
        std_dev_activation_large = torch.std(tensor_large)
        std_dev_activation_small = torch.std(tensor_small)

        sparsity = (tensor == 0).float().mean() * 100 # Percentage of zeros in the tensor
        max_activation = torch.max(tensor)
        min_activation = torch.min(tensor)
       
        saturation_threshold = 0.01 
        saturated_neurons = ((tensor < saturation_threshold) | (tensor > (1 - saturation_threshold))).float().mean() * 100

        unique_elements_large, counts_large = torch.unique(tensor_large, return_counts=True)
        max_count_indices_large = torch.where(counts_large == torch.max(counts_large))[0]
        mode_tensor_large = unique_elements[max_count_indices_large]

        unique_elements_small, counts_small = torch.unique(tensor_small, return_counts=True)
        max_count_indices_small = torch.where(counts_small == torch.max(counts_small))[0]
        mode_tensor_small = unique_elements[max_count_indices_small]



        self.large_tensors_index = torch.cat(self.large_tensors_index, indices_large)
        self.small_tensors_index = torch.cat(self.small_tensors_index, indices_small)
        self.large_tensors = torch.cat(self.large_tensors, tensor_large)
        self.small_tensors = torch.cat(self.small_tensors, tensor_small)

        self.means.append(mean_activation)
        self.means_large.append(mean_activation_large)
        self.means_small.append(mean_activation_small)

        self.std_dev.append(std_dev_activation)
        self.std_dev_large.append(std_dev_activation_large)
        self.std_dev_small.append(std_dev_activation_small)

        self.sparsity.append(sparsity)

        self.max_activations.append(max_activation)
        self.min_activations.append(min_activation)

        self.saturated_neurons.append(saturated_neurons)
        self.large_mode.append(mode_tensor_large)
        self.small_mode.append(mode_tensor_small)



