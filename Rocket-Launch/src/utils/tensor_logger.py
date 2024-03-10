import torch.nn as nn
import torch
import os
import plotly.express as px
import pandas as pd
import numpy as np

# Must run pip install plotly and pip install -U kaleido


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
        self.large_modes = []
        self.small_modes = []

        # Must make sure this directory exists, I was thinking we could create an experiment name field
        # in the config file and then use it to store our results more easily.
        self.base_output_path = "/grphome/grp_inject/compute/logging"
        self.experiment_name = "test"  # config.experiment_name?

        self.layer_numbers = []
        
        

    def addTensor(self, tensor: torch.Tensor):
        tensor = tensor.flatten()
        
        # For use by the heatmap generator
        self.layer_numbers = [i for i in range(len(tensor))]

        # Store the largest 1000 values in each tensor and their indices
        tensor_large, indices_large = torch.topk(tensor, 1000, largest=True)
        tensor_small, indices_small = torch.topk(tensor, 1000, largest=False)
        
        # Store the mean activation for each tensor
        mean_activation = torch.mean(tensor)
        mean_activation_large = torch.mean(tensor_large)
        mean_activation_small = torch.mean(tensor_small)

        # Store the standard deviation for each tensor
        std_dev_activation = torch.std(tensor)
        std_dev_activation_large = torch.std(tensor_large)
        std_dev_activation_small = torch.std(tensor_small)

        # Store the sparcity, maximum activation and minimum activation
        sparsity = (tensor == 0).float().mean() * 100 # Percentage of zeros in the tensor
        max_activation = torch.max(tensor)
        min_activation = torch.min(tensor)
       
        # Calculate the percentage of saturated neurons?  Not sure what saturation means
        saturation_threshold = 0.01 
        saturated_neurons = ((tensor < saturation_threshold) | (tensor > (1 - saturation_threshold))).float().mean() * 100

        k = 10
        # Finding the k highest frequencies of the large tensor (most_frequent_values_large[0] is the mode)
        unique_elements_large, counts_large = torch.unique(tensor_large, return_counts=True)
        top_counts_large_values, top_counts_large_indices = torch.topk(counts_large, k)
        most_frequent_values_large = unique_elements_large[top_counts_large_indices]

        # Finding the k highest frequencies of the small tensor (most_frequent_values_small[0] is the mode)
        unique_elements_small, counts_small = torch.unique(tensor_small, return_counts=True)
        top_counts_small_values, top_counts_small_indices = torch.topk(counts_small, k)
        most_frequent_values_small = unique_elements_large[top_counts_small_indices]


        self.large_tensors_index = torch.cat((self.large_tensors_index, indices_large.unsqueeze(0)), dim=0)
        self.small_tensors_index = torch.cat((self.small_tensors_index, indices_small.unsqueeze(0)), dim=0)
        self.large_tensors = torch.cat((self.large_tensors, tensor_large.unsqueeze(0)), dim=0)
        self.small_tensors = torch.cat((self.small_tensors, tensor_small.unsqueeze(0)), dim=0)


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
        self.large_modes.append(most_frequent_values_large)
        self.small_modes.append(most_frequent_values_small)


    def write_log(self):
        # Ensure the directory exists
        directory_path = os.path.join(self.base_output_path, self.experiment_name)
        os.makedirs(directory_path, exist_ok=True)
        
        # Specify the filename for your log file
        log_filename = "experiment_log.txt"  # Name of the log file
        
        # Full path for the log file
        log_file_path = os.path.join(directory_path, log_filename)
        
        # Now, open the file to append the logs
        with open(log_file_path, 'a') as f:
            f.write("Current model weights means: {}\n".format(self.means))
            f.write("Means of the top 1000 weights: {}\n".format(self.means_large))
            f.write("Means of the bottom 1000 weights: {}\n\n".format(self.means_small))

            f.write("Current model weights standard deviation: {}\n".format(self.std_dev))
            f.write("Standard deviation of the top 1000 weights: {}\n".format(self.std_dev_large))
            f.write("Standard deviation of the bottom 1000 weights: {}\n\n".format(self.std_dev_small))

            f.write("Current model weights sparsity: {}\n\n".format(self.sparsity))

            f.write("Current model maximum activation: {}\n".format(self.max_activations))
            f.write("Current model minimum activation: {}\n\n".format(self.min_activations))

            f.write("Current model saturation: {}\n".format(self.saturated_neurons))
            f.write("Top 10 frequent values of the top 1000 weights: {}\n".format(self.large_modes))
            f.write("Top 10 frequent values of the bottom 1000 weights: {}\n".format(self.small_modes))


    # Generates a heatmap showing the locations of the largest and the smallest weights for the layers.  Will require some additional packages to be installed.
    def generate_heatmap(self):
        images_dir = os.path.join(self.base_output_path, self.experiment_name, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Convert PyTorch tensors to NumPy arrays, detaching them from the computation graph
        large_tensor_index_np = self.large_tensors_index.cpu().detach().numpy()
        large_tensor_index_np = large_tensor_index_np.astype(int)
        large_tensor_value_np = self.large_tensors.cpu().detach().numpy()

        # Assuming you know the original shape of the data, 'dims'
        dims = (4000, 4000)  # Example, adjust to your actual dimensions

        # Create an empty 2D array for the heatmap
        heatmap_data = np.zeros(dims)

        # Fill in the heatmap data using the indices and values
        for idx, value in zip(large_tensor_index_np, large_tensor_value_np):
            row, col = np.unravel_index(idx, dims)
            heatmap_data[row, col] = value

        # Generate and save the heatmap
        fig = px.imshow(heatmap_data, color_continuous_scale='Viridis', labels={'color': 'Value'})
        fig.update_layout(title="Large Tensor Values Heatmap")
        fig.write_image(os.path.join(images_dir, "large_tensor_heatmap.png"))
        
        # os.makedirs(os.path.join(self.base_output_path, self.experiment_name, "images"))
        # large_df = pd.DataFrame(zip(self.large_tensors_index, self.large_tensors), columns=['Index', 'Value'])
        # fig = px.imshow(large_df, x='Index', y='Value')
        # fig.write_image(os.path.join(self.base_output_path, self.experiment_name, "images/large_heatmap.png"))

        # small_df = pd.DataFrame(zip(self.small_tensors_index, self.small_tensors), columns=['Index', 'Value'])
        # fig_1 = px.imshow(small_df, x='Index', y='Value')
        # fig_1.write_image(os.path.join(self.base_output_path, self.experiment_name, "images/small_heatmap.png"))
