import torch.nn as nn
import torch
import os
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# Must run pip install plotly and pip install -U kaleido

# important data:
# layers heatmap that shows most activated parts
# bar graph showing most activated layers
# sparsity graph showing which layers are heavily activated
# graph showing which parts of a paticular layer is important? 


class tensor_logger:
    
	# TODO: Implement a part of the logger that will take the average across a whole prompt and write a file
    # TODO: Implement new config fields for logging (DONE)
    # TODO: Add logging functionality to inference loop
    # TODO: Clean up the logging code 

    def __init__(self, num_hidden_layers):
        self.num_hidden_layers = num_hidden_layers

        self.store_prompt_values = torch.empty(0)
        self.store_prompt_indices = torch.empty(0)
        self.tensor_length = 0
        self.token_number = 1

        self.prompt_df = pd.DataFrame()

        self.layered_tensor = torch.empty(0)
        self.all = torch.empty(0)

        self.means = torch.empty(0)

        self.std_dev = torch.empty(0)

        self.sparsity = torch.empty(0)

        self.max_activations = torch.empty(0)
        self.min_activations = torch.empty(0)

        self.modes = torch.empty(0)

        # Must make sure this directory exists, I was thinking we could create an experiment name field
        # in the config file and then use it to store our results more easily.
        self.base_output_path = "/grphome/grp_inject/compute/logging"
        self.experiment_name = "/test/"  # config.experiment_name?


    def new_prompt(self):
        indices = self.make_layer_indicies(self.store_prompt_indices.flatten()).detach().numpy()
        values = self.store_prompt_values.flatten().detach().numpy()
        layers = self.assign_layer(self.store_prompt_indices.flatten()).detach().numpy()

        self.prompt_df =  pd.DataFrame({'index': indices, 'value': values, 'layer': layers})
        self.prompt_df = self.prompt_df.groupby(['index', 'layer'], as_index=False)['value'].mean()

        print(self.prompt_df)
        
        name_of_csv = f'index_value_layer_{self.token_number}.csv'
        self.prompt_df.to_csv(self.base_output_path + self.experiment_name + name_of_csv)

        self.generate_index_value_layer_heatmap()

        self.store_prompt_values = torch.empty(0)
        self.store_prompt_indices = torch.empty(0)
        self.prompt_df = pd.DataFrame()
        self.token_number += 1

    def assign_layer(self, tensor):
        tensor_divisor = self.tensor_length // self.num_hidden_layers

        return (tensor // tensor_divisor) + 1

    def make_layer_indicies(self, tensor):
        tensor_divisor = self.tensor_length // self.num_hidden_layers

        return tensor % tensor_divisor
        
        
    def add_tensor(self, tensor: torch.Tensor):
        
        tensor = tensor.flatten()
        self.map_layers(tensor)

        self.all = tensor
        self.tensor_length = self.all.size(0)

        # For use by the heatmap generator
        self.layer_numbers = [i for i in range(len(tensor))]

        # Store the largest 1000 values in each tensor and their indices
        tensor_large, indices_large = torch.topk(tensor, 1000, largest=True)

        # Store the mean activation for each tensor
        mean_activation = self.layered_tensor.mean(dim=1)

        # Store the standard deviation for each tensor
        std_dev_activation = self.layered_tensor.std(dim=1)

        num_zeros = self.layered_tensor.eq(0).sum(dim=1)  # Count zeros in each row
        total_elements_per_subtensor = self.layered_tensor.size(1)  # Number of elements per row
        sparsity = num_zeros.float() / total_elements_per_subtensor

        max_activation = self.layered_tensor.max(dim=1).values 
        min_activation = self.layered_tensor.min(dim=1).values 

        modes = torch.empty(self.layered_tensor.size(0), dtype=self.layered_tensor.dtype)

        for i in range(self.layered_tensor.size(0)):
            sub_tensor = self.layered_tensor[i].view(-1)  # Flatten the sub-tensor
            unique_values, counts = sub_tensor.unique(return_counts=True)
            modes[i] = unique_values[counts.argmax()]


        self.store_prompt_values = torch.cat((self.store_prompt_values, tensor_large.unsqueeze(0)), dim=0)
        self.store_prompt_indices = torch.cat((self.store_prompt_indices, indices_large.unsqueeze(0)), dim=0)
    
        torch.cat((self.means, mean_activation), dim=0)
        torch.cat((self.std_dev, std_dev_activation), dim=0)
        torch.cat((self.sparsity, sparsity), dim=0)

        torch.cat((self.max_activations, max_activation), dim=0)
        torch.cat((self.min_activations, min_activation), dim=0)

        torch.cat((self.modes, modes), dim=0)

    def map_layers(self, tensor: torch.Tensor):
        divided_tensors = tensor.chunk(self.num_hidden_layers, dim=0)

        self.layered_tensor = torch.stack(divided_tensors)
        print(self.layered_tensor.dim())

        divided_tensors = [t.squeeze(0) for t in divided_tensors]

        self.heatmap_data = torch.cat(divided_tensors, dim=0)

        return self.layered_tensor

        # self.map_layers = {
        #     layer: [(i, divided_tensors[layer - 1][i]) for i in range(len(divided_tensors[layer - 1]))]  # 1000 important weights per layer
        #     for layer in range(1, self.num_hidden_layers + 1)
        # }


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

            f.write("Current model weights standard deviation: {}\n".format(self.std_dev))
            f.write("Standard deviation of the top 1000 weights: {}\n".format(self.std_dev_large))

            f.write("Current model weights sparsity: {}\n\n".format(self.sparsity))

            f.write("Current model maximum activation: {}\n".format(self.max_activations))
            f.write("Current model minimum activation: {}\n\n".format(self.min_activations))

            f.write("Top 10 frequent values of the top 1000 weights: {}\n".format(self.modes))


    # Generates a heatmap showing the locations of the largest and the smallest weights for the layers.  Will require some additional packages to be installed.
    def generate_heatmap(self):
        # images_dir = os.path.join(self.base_output_path, self.experiment_name, "images")
        # os.makedirs(images_dir, exist_ok=True)
        
        # # Convert PyTorch tensors to NumPy arrays, detaching them from the computation graph
        # large_tensor_index_np = self.large_tensors_index.cpu().detach().numpy()
        # large_tensor_index_np = large_tensor_index_np.astype(int)
        # large_tensor_value_np = self.large_tensors.cpu().detach().numpy()

        # # Assuming you know the original shape of the data, 'dims'
        # dims = (4000, 4000)  # Example, adjust to your actual dimensions

        # # Create an empty 2D array for the heatmap
        # layered_tensor = np.zeros(dims)

        # # Fill in the heatmap data using the indices and values
        # for idx, value in zip(large_tensor_index_np, large_tensor_value_np):
        #     row, col = np.unravel_index(idx, dims)
        #     layered_tensor[row, col] = value

        # # Generate and save the heatmap
        # fig = px.imshow(layered_tensor, color_continuous_scale='Viridis', labels={'color': 'Value'})
        # fig.update_layout(title="Large Tensor Values Heatmap")
        # fig.write_image(os.path.join(images_dir, "large_tensor_heatmap.png"))

        
        # # Convert PyTorch tensors to NumPy arrays, detaching them from the computation graph
        # small_tensor_index_np = self.small_tensors_index.cpu().detach().numpy()
        # small_tensor_index_np = small_tensor_index_np.astype(int)
        # small_tensor_value_np = self.small_tensors.cpu().detach().numpy()

        # # Assuming you know the original shape of the data, 'dims'
        # dims = (4000, 4000)  # Example, adjust to your actual dimensions

        # # Create an empty 2D array for the heatmap
        # layered_tensor = np.zeros(dims)

        # # Fill in the heatmap data using the indices and values
        # for idx, value in zip(small_tensor_index_np, small_tensor_value_np):
        #     row, col = np.unravel_index(idx, dims)
        #     layered_tensor[row, col] = value

        # # Generate and save the heatmap
        # fig = px.imshow(layered_tensor, color_continuous_scale='Viridis', labels={'color': 'Value'})
        # fig.update_layout(title="Small Tensor Values Heatmap")
        # fig.write_image(os.path.join(images_dir, "small_tensor_heatmap.png"))
        
        # os.makedirs(os.path.join(self.base_output_path, self.experiment_name, "images"))
        # large_df = pd.DataFrame(zip(self.large_tensors_index, self.large_tensors), columns=['Index', 'Value'])
        # fig = px.imshow(large_df, x='Index', y='Value')
        # fig.write_image(os.path.join(self.base_output_path, self.experiment_name, "images/large_heatmap.png"))

        # small_df = pd.DataFrame(zip(self.small_tensors_index, self.small_tensors), columns=['Index', 'Value'])
        # fig_1 = px.imshow(small_df, x='Index', y='Value')
        # fig_1.write_image(os.path.join(self.base_output_path, self.experiment_name, "images/small_heatmap.png"))
        # Reshape the data for 2D visualization

        # Assuming self.layered_tensor is a numpy array and self.num_hidden_layers is defined
        # num_indices_per_layer is calculated as shown
        layered_tensor = self.heatmap_data.detach().numpy()
        num_indices_per_layer = len(layered_tensor) // self.num_hidden_layers

        # Reshape the data for 2D visualization
        tensor_2d = layered_tensor.reshape(self.num_hidden_layers, num_indices_per_layer)

        # Flatten the 2D tensor to work with 1D arrays for identifying top values
        flat_data = tensor_2d.flatten()
        # Get indices of the top 1000 values
        top_1000_indices = np.argpartition(flat_data, -1000)[-1000:]
        # Convert 1D indices to 2D indices for plotting
        rows, cols = np.unravel_index(top_1000_indices, tensor_2d.shape)

        # Normalization for the heatmap
        vmin = np.min(tensor_2d)
        vmax = np.max(tensor_2d)  # or set a specific value to focus on a range
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Plotting the base heatmap
        plt.figure(figsize=(15, 10))
        plt.imshow(tensor_2d, cmap='viridis', aspect='auto', norm=norm)
        plt.colorbar(label='Value')

        # Overlay the top 1000 values with a distinct marker
        # Customize the marker style ('o', '*', etc.), size, color, and edgecolor as needed
        plt.scatter(cols, rows, color='red', s=10, edgecolor='white', marker='o', label='Top 1000 Values')

        plt.title('Heatmap of Layer Index Values Highlighting Top 1000 Values')
        plt.xlabel('Index')
        plt.ylabel('Layer')
        plt.yticks(np.arange(self.num_hidden_layers), np.arange(1, self.num_hidden_layers + 1))
        plt.legend()  # Add a legend if needed
        plt.show()

        # Save the figure to a file
        plt.savefig('/grphome/grp_inject/compute/logging/test/images/heatmap_mean_layer_values.png')  # Adjust path as needed

    def generate_index_value_layer_heatmap(self):
        pivot_df = self.prompt_df.pivot(index="layer", columns="index", values="value")
        pivot_df_filled = pivot_df.fillna(-1)

        heatmap_file_name = f'index_layer_value_heatmap_{self.token_number}.png'

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        # Use a colormap (cmap) that distinguishes your filled value (-1) from the rest, e.g., 'coolwarm'
        # You may need to adjust the colormap or the fill value depending on your data range and preferences
        sns.heatmap(pivot_df_filled, cmap="coolwarm", annot=True, cbar_kws={'label': 'Value'}, 
                    norm=plt.Normalize(vmin=pivot_df_filled.min().min(), vmax=pivot_df_filled.max().max()))

        plt.title('Heatmap of Values')
        plt.xlabel('Layer')
        plt.ylabel('Index')
        plt.show()

        plt.savefig(self.base_output_path + self.experiment_name + 'images/' + heatmap_file_name)

    def generate_histograms(self):

        num_indices_per_layer = len(self.layered_tensor) // self.num_hidden_layers
        values, indices = torch.topk(self.all, 1000)

        # Determine which layer each top value belongs to
        layer_indices = (indices / num_indices_per_layer).floor().long()

        # Count the occurrences of top values in each layer
        layer_counts = np.bincount(layer_indices.numpy(), minlength=self.num_hidden_layers)

        # Generate the bar chart
        layers = np.arange(1, self.num_hidden_layers + 1)
        plt.figure(figsize=(10, 6))
        plt.bar(layers, layer_counts, color='skyblue')
        plt.xlabel('Layer')
        plt.ylabel('Count of Top 1000 Values')
        plt.title('Top 1000 Values Distribution Across Layers')
        plt.xticks(layers)
        plt.show()

        plt.savefig('/grphome/grp_inject/compute/logging/test/images/bar_graph_of_largest_by_layer.png') 
        pass

    def hard_coded_graph(self):
        layer_counts_1 = np.array([90, 85, 80, 95, 100, 105, 110, 105, 100, 95, 80, 55])  # Example distribution that sums to 1000
        layer_counts_2 = np.array([55, 80, 95, 100, 105, 110, 105, 100, 95, 80, 85, 90]) 

        num_layers = len(layer_counts_1)  # Assuming both tensors have counts for the same number of layers
        layers = np.arange(1, num_layers + 1)
        bar_width = 0.35  # Width of the bars

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plotting both distributions
        bar1 = ax.bar(layers - bar_width/2, layer_counts_1, bar_width, label='Anger Alignment IRM', color='skyblue')
        bar2 = ax.bar(layers + bar_width/2, layer_counts_2, bar_width, label='Cheerful Alignment IRM', color='orange')

        # Add some text for labels, title, and custom x-axis tick labels, etc.
        ax.set_xlabel('Layer')
        ax.set_ylabel('Count of Top 1000 Values')
        ax.set_title('Comparison of Top 1000 Values Distribution Across Layers')
        ax.set_xticks(layers)
        ax.set_xticklabels([f'Layer {i}' for i in layers])
        ax.legend()

        # Finally, show the plot
        plt.show()

        plt.savefig('/grphome/grp_inject/compute/logging/test/images/compare_different_alignments_hard_coded.png') 

    def sparcity_graph(self):
        pass

        
