import torch.nn as nn
import torch
import os
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import seaborn as sns
from PIL import Image
from natsort import natsorted

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

    def __init__(self, num_hidden_layers, experiment_name, layers):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Determine device
        self.num_hidden_layers = num_hidden_layers
        self.layers = layers

        self.fig = plt.figure(figsize=(16, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.store_prompt_values = torch.empty(0).to(self.device)
        self.store_prompt_indices = torch.empty(0).to(self.device)
        self.tensor_length = 0
        self.token_number = 1
        self.prompt_number = 1

        self.prompt_df = pd.DataFrame()

        self.layered_tensor = torch.empty(0).to(self.device)
        self.all = torch.empty(0)

        self.means = torch.empty(0).to(self.device)

        self.std_dev = torch.empty(0).to(self.device)

        self.sparsity = torch.empty(0).to(self.device)

        self.max_activations = torch.empty(0).to(self.device)
        self.min_activations = torch.empty(0).to(self.device)

        self.modes = torch.empty(0).to(self.device)

        self.base_output_path = "/grphome/grp_inject/compute/logging"
        self.experiment_name = experiment_name  # config.experiment_name?
        os.makedirs(os.path.join(self.base_output_path, self.experiment_name), exist_ok=True)

    def new_prompt(self):
		# ## TODO: Look at this code ##

        # self.prompt_df =  pd.DataFrame({'index': indices, 'value': values, 'layer': layers})
        # #print(self.prompt_df, flush=True)
        # self.prompt_df = self.prompt_df.groupby(['index', 'layer'], as_index=False)['value'].mean()
        # #print(self.prompt_df, flush=True)

        # #print(self.prompt_df)
        
        # name_of_csv = f'index_value_layer_{self.token_number}.csv'
        # #self.prompt_df.to_csv(os.path.join(self.base_output_path,self.experiment_name,name_of_csv), index=False)

        # #self.generate_index_value_layer_heatmap()

        # self.store_prompt_values = torch.empty(0).to(self.device)
        # self.store_prompt_indices = torch.empty(0).to(self.device)
        # self.prompt_df = pd.DataFrame()
        self.token_number = 1
        self.prompt_number += 1
        
    def get_layer_weights(self, tensor, layer_id):
        return tensor[:, :, :, self.layers.index(layer_id)]
    
    def write_csv(self, df):
        name_of_csv = f'index_value_layer_{self.token_number}.csv'
        self.output_path = os.path.join(self.base_output_path, self.experiment_name, "Prompt{}_CSVs".format(self.prompt_number))
        os.makedirs(self.output_path, exist_ok=True)
        df.to_csv(os.path.join(self.output_path,name_of_csv), index=False)
        
    def add_tensor(self, tensor: torch.Tensor):
        ## ROADMAP ##
        # 1. Split tensor into layers
        # 1.5. Divide up by sequence length? 
        # 2. Flatten the tensor
        # 3. Make heatmap boiz - at each individual token and averaged across the prompt

        print(tensor.shape[1], flush=True)
        if tensor.shape[1] > 1:
            print("Splitting tensor", flush=True)
            tensors = torch.split(tensor, 1, dim=1)
            for tensor in tensors:
                self.add_tensor(tensor)
        
        weights = torch.flatten(tensor, start_dim=0, end_dim=2).cpu().detach().numpy()
        
        print("Weights shape: {}\n\n".format(weights.shape), flush=True)
        #print("Weights: {}\n\n".format(weights.head()), flush=True)
        print("Weights Length: {}\n\n".format(len(weights)), flush=True)
        dataFrames = []
        for i in range(weights.shape[1]):
            curr_weights = weights[:, i]
            curr_layer = self.layers[i]
            dataFrames.append(pd.DataFrame({'index': range(len(curr_weights)),  'layer': [curr_layer for j in range(len(curr_weights))], 'value': curr_weights}))
        self.token_df =  pd.concat(dataFrames)


        self.write_csv(self.token_df)
        #self.generate_index_value_layer_heatmap()
        self.token_number += 1


        ## TODO: CHECK THIS CODE! ##
        #self.tensor_length = self.all.size(0)
        #print("Tensor length: {}".format(self.tensor_length), flush=True)

        # For use by the heatmap generator
        #self.layer_numbers = [i for i in range(len(tensor))]

        # Store the largest 1000 values in each tensor and their indices
        # tensor_large, indices_large = torch.topk(tensor, 1000, largest=True)

        # # Store the mean activation for each tensor
        # mean_activation = self.layered_tensor.mean(dim=1)

        # # Store the standard deviation for each tensor
        # std_dev_activation = self.layered_tensor.std(dim=1)

        # num_zeros = self.layered_tensor.eq(0).sum(dim=1)  # Count zeros in each row
        # total_elements_per_subtensor = self.layered_tensor.size(1)  # Number of elements per row
        # sparsity = num_zeros.float() / total_elements_per_subtensor

        # max_activation = self.layered_tensor.max(dim=1).values 
        # min_activation = self.layered_tensor.min(dim=1).values 

        # modes = torch.empty(self.layered_tensor.size(0), dtype=self.layered_tensor.dtype).to(self.device)

        # for i in range(self.layered_tensor.size(0)):
        #     sub_tensor = self.layered_tensor[i].view(-1)  # Flatten the sub-tensor
        #     unique_values, counts = sub_tensor.unique(return_counts=True)
        #     modes[i] = unique_values[counts.argmax()]


        # self.store_prompt_values = torch.cat((self.store_prompt_values, tensor_large.unsqueeze(0)), dim=0)
        # self.store_prompt_indices = torch.cat((self.store_prompt_indices, indices_large.unsqueeze(0)), dim=0)
    
        # torch.cat((self.means, mean_activation), dim=0)
        # torch.cat((self.std_dev, std_dev_activation), dim=0)
        # torch.cat((self.sparsity, sparsity), dim=0)

        # torch.cat((self.max_activations, max_activation), dim=0)
        # torch.cat((self.min_activations, min_activation), dim=0)

        # torch.cat((self.modes, modes), dim=0)


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
            f.write("Current model weights means:\n")
            f.write(self.means.cpu().__str__())
           # f.write("Means of the top 1000 weights: {}\n".format(self.means_large))

            f.write("Current model weights standard deviation:\n")
            f.write(np.array2string(self.std_dev.cpu().detach().numpy()))
            #f.write("Standard deviation of the top 1000 weights: {}\n".format(self.std_dev_large))

            f.write("Current model weights sparsity:\n")
            f.write(np.array2string(self.sparsity.cpu().detach().numpy()))

            f.write("Current model maximum activation: {}\n".format(np.array2string(self.max_activations.cpu().detach().numpy())))
            f.write("Current model minimum activation: {}\n\n".format(np.array2string(self.min_activations.cpu().detach().numpy())))

            f.write("Top 10 frequent values of the top 1000 weights: {}\n".format(np.array2string(self.modes.cpu().detach().numpy())))

    def generate_frame(self, num, dataframes, pivoted_dfs):
    # Generates a single frame of the animation
        self.fig.clf()  # Clear the previous figure
        self.ax = self.fig.add_subplot(111, projection='3d')

        df = pivoted_dfs[num]  # Get data for the current frame 

        # Plotting logic (similar to before)
        x_pos, y_pos = np.meshgrid(df.columns, df.index)
        x_pos = x_pos.flatten()
        y_pos = y_pos.flatten()
        z_pos = np.zeros_like(x_pos)  # 0 on the 'floor'

        # Get the bar heights and colors
        dz = df.to_numpy().flatten()  
        color_map = plt.cm.get_cmap('viridis')  # Choose a colormap

        # Create the 3D bars
        self.ax.bar3d(x_pos, y_pos, z_pos, dx=0.5, dy=0.5, dz=dz, color=color_map(dz))

        

         # Create a smaller axes for the colorbar
        cax = self.fig.add_axes([0.9, 0.1, 0.03, 0.8])  # Adjust position as needed

        # Create a scalar mappable for the legend
        norm = plt.Normalize(self.min_value, self.max_value)  # Normalize colors 
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])  # This is necessary to avoid a potential bug

        # Add the colorbar and its label
        cbar = self.fig.colorbar(sm, cax=cax, label="Value") 

        # Labels and adjustments
        #self.ax.set_ylim3d(0, len(self.layers))
        self.ax.set_zlim3d(self.min_value, self.max_value)
        self.ax.set_xlabel('Layer')
        self.ax.set_ylabel('Index')
        self.ax.set_zlabel('Value')
        self.ax.set_title("3D Heatmap {}".format(num))

        self.fig.canvas.draw()  # Draw the current frame

        # Convert the matplotlib figure to a Pillow Image
        img = Image.frombytes('RGB', self.fig.canvas.get_width_height(),
                            self.fig.canvas.tostring_rgb())
        return img  # Return the Pillow Image


    def generate_anim(self, data):

        pivoted_dfs = []
        for df in data:

            pivoted_df = df.pivot(index='index', columns='layer', values='value')
            pivoted_dfs.append(pivoted_df)

        # Get positions for the bars
        x_pos, y_pos = np.meshgrid(pivoted_df.columns, pivoted_df.index)
        x_pos = x_pos.flatten()
        y_pos = y_pos.flatten()
        z_pos = np.zeros_like(x_pos)  # 0 on the 'floor'

        # Get the bar heights and colors
        dz = pivoted_df.to_numpy().flatten()  
        color_map = plt.cm.get_cmap('viridis')  # Choose a colormap

        # Create the 3D bars
        self.ax.bar3d(x_pos, y_pos, z_pos, dx=0.5, dy=0.5, dz=dz, color=color_map(dz))

    

        # Labels and adjustments
        self.ax.set_xlabel('Layer')
        self.ax.set_ylabel('Index')
        self.ax.set_zlabel('Value')
        self.ax.set_title("3D Heatmap ")

        frames = []  # Store the generated frames as Pillow Images
        for num in range(len(pivoted_dfs)):
            frames.append(self.generate_frame(num, data, pivoted_dfs))

        output_path = os.path.join(self.base_output_path, self.experiment_name, 'images/', "prompt_{}".format(self.prompt_number))
        os.makedirs(output_path, exist_ok=True)
        # Save the animation
        frames[0].save(os.path.join(output_path, "heatmap_animation{}.gif".format(self.prompt_number)), format='GIF', append_images=frames[1:], 
                    save_all=True, duration=200, loop=0)


    def recursively_generate_heatmap(self, path):
        data = []
        csv_identifiers = []
        csv_counter = 1
        if os.path.isdir(path):
            print("Path: ", path, flush=True)
            files = os.listdir(path)

            # Sort files using natural sorting
            files = natsorted(files) 

            for file in files:
                if os.path.isdir(os.path.join(path, file)):
                    self.recursively_generate_heatmap(os.path.join(path, file)) 
                else:
                    print("File: ", file, flush=True)
                    df = pd.read_csv(os.path.join(path, file))
                    df['csv_number'] = csv_counter
                    data.append(df)
                    csv_identifiers.append(csv_counter)
                    csv_counter += 1

        self.generate_histograms(data)
        self.calculate_and_plot_sparsity(data)
        self.create_histogram_of_top_values_by_csv(data, csv_identifiers)
        self.generate_average_histograms(data)
        

        i = 1
        for df in data:
            print("Dataframe{}:\n".format(i), flush = True)
            print(df.head(), flush = True)
            max_index = df['value'].idxmax()
            print("Max value and index: ", df.loc[max_index], flush = True)
            min_index = df['value'].idxmin()
            print("Min value and index: ", df.loc[min_index], flush = True)
            i+=1
        self.max_value = max([df['value'].max() for df in data])
        print("Max value:{}".format(self.max_value), flush = True)
        self.min_value = min([df['value'].min() for df in data])
        print("Min Value:{}".format(self.min_value), flush = True)
        
        i = 1

        ### IF YOU WANT TO GENERATE A 3D HEATMAP, UNCOMMENT THE FOLLOWING LINE ###

        print("Generating 3D heatmap", flush = True)
        # self.generate_anim(data)

        for df in data:
            pivot_df = df.pivot(index="layer", columns="index", values="value")
            print("Pivoted dataframe:\n", flush = True)
            print(pivot_df.head(), flush = True)
            #pivot_df = pivot_df.iloc[:self.num_hidden_layers] 
            pivot_df_filled = pivot_df.fillna(0)
            #print(pivot_df_filled, flush=True)

            heatmap_file_name = f'index_layer_value_heatmap_{i}.png'
            print("\n\n\n MIN : {}\n\n\n".format(pivot_df.min().min()), flush = True)
            # Create the heatmap
            plt.figure(figsize=(14, 8))
            # Use a colormap (cmap) that distinguishes your filled value (-1) from the rest, e.g., 'coolwarm'
            # You may need to adjust the colormap or the fill value depending on your data range and preferences
            sns.heatmap(pivot_df_filled, cmap="coolwarm", cbar_kws={'label': 'Value'}, vmin=pivot_df.min().min(), vmax=pivot_df.max().max()) # norm=plt.Normalize(vmin=pivot_df_filled.min().min(), vmax=pivot_df_filled.max().max()

            plt.title('Heatmap of Values')
            plt.ylabel('Layer')
            plt.xlabel('Index')
            plt.show()
            output_path = os.path.join(self.base_output_path, self.experiment_name, 'images/', "prompt_{}".format(self.prompt_number))
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, heatmap_file_name))
            plt.close()
            i += 1

        result = pd.concat(data).groupby(['layer', 'index'])['value'].mean().reset_index()
        print("Result:\n", flush = True)
        print(result.head(), flush = True)
        pivot_df = result.pivot(index="layer", columns="index", values="value")
        #pivot_df = result.unstack()
        print("Pivoted final dataframe:\n", flush = True)
        print(pivot_df.head(), flush = True)
        pivot_df = pivot_df.iloc[:self.num_hidden_layers] 
        pivot_df_filled = pivot_df.fillna(0)
        #print(pivot_df_filled, flush=True)

        heatmap_file_name = f'Average_value_heatmap_{self.token_number}.png'

        # Create the heatmap
        plt.figure(figsize=(14, 8))
        print("\n\n\n MIN : {}\n\n\n".format(pivot_df.min()), flush = True)
        # Use a colormap (cmap) that distinguishes your filled value (-1) from the rest, e.g., 'coolwarm'
        # You may need to adjust the colormap or the fill value depending on your data range and preferences
        sns.heatmap(pivot_df_filled, cmap="coolwarm", cbar_kws={'label': 'Value'}, vmin=pivot_df.min().min(), vmax=pivot_df.max().max()) # norm=plt.Normalize(vmin=pivot_df_filled.min().min(), vmax=pivot_df_filled.max().max()

        plt.title('Heatmap of Average Values')
        plt.ylabel('Layer')
        plt.xlabel('Index')
        plt.show()
        #os.makedirs(os.path.join(self.base_output_path, self.experiment_name, 'images/', "prompt_{}".format(self.prompt_number)), exist_ok=True)
        plt.savefig(os.path.join(output_path, heatmap_file_name))


    def generate_heatmap_1(self):
        self.recursively_generate_heatmap(self.output_path)
        

    # Generates a heatmap showing the locations of the largest and the smallest weights for the layers.  Will require some additional packages to be installed.
    def generate_heatmap(self):

        layered_tensor = self.heatmap_data.cpu().detach().numpy()
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
        plt.scatter(cols, rows, color='red', s=2, edgecolor='white', marker='o', label='Top 1000 Values')

        plt.title('Heatmap of Layer Index Values Highlighting Top 1000 Values')
        plt.xlabel('Index')
        plt.ylabel('Layer')
        plt.yticks(np.arange(self.num_hidden_layers), np.arange(1, self.num_hidden_layers + 1))
        plt.legend()  # Add a legend if needed
        plt.show()

        # Save the figure to a file
        plt.savefig(os.path.join(self.base_output_path, self.experiment_name, "images/test_heatmap{}.png".format(self.token_number)))  # Adjust path as needed

    def generate_histograms(self, data):
        combined_df = pd.concat(data)

        combined_df['abs_value'] = combined_df['value'].abs()
        top_2000 = combined_df.nlargest(100, 'abs_value')

        freq_distribution = top_2000['layer'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        freq_distribution.plot(kind='bar')
        plt.title('Frequency of Top 100 Absolute Values by Layer')
        plt.xlabel('Layer')
        plt.ylabel('Frequency')
        
        plt.savefig(os.path.join(self.base_output_path, self.experiment_name, "images/histogram_per_layer.png"))
        
    def generate_average_histograms(self, data):
        grouped_data = [data[i:i + 4] for i in range(0, len(data), 4)]
        group_counter = 1
        for group in grouped_data:
            combined_df = pd.concat(group).groupby(['layer', 'index'], as_index=False).mean()
            pivot_df = combined_df.pivot(index="layer", columns="index", values="value").fillna(0)

            plt.figure(figsize=(14, 8))
            sns.heatmap(pivot_df, cmap="coolwarm", cbar_kws={'label': 'Value'}, vmin=pivot_df.min().min(), vmax=pivot_df.max().max())
            plt.title(f'Heatmap of Average Values for Group {group_counter}')
            plt.ylabel('Layer')
            plt.xlabel('Index')

            heatmap_file_name = f'average_heatmap_group_{group_counter}.png'
            output_path = os.path.join(self.base_output_path, self.experiment_name, 'images', f"prompt_{self.prompt_number - 1}")
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, heatmap_file_name))
            plt.close()

            group_counter += 1
        
    def create_histogram_of_top_values_by_csv(self, data, csv_identifiers):
        combined_df = pd.concat(data)

        combined_df['abs_value'] = combined_df['value'].abs()

        top_2000 = combined_df.nlargest(1000, 'abs_value')
        freq_distribution_by_csv = top_2000['csv_number'].value_counts().reindex(csv_identifiers, fill_value=0)

        # Plotting the histogram
        plt.figure(figsize=(12, 7))
        freq_distribution_by_csv.plot(kind='bar')
        plt.title('Frequency of Overall Top 1000 Absolute Values by Token')
        plt.xlabel('CSV Number')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to not cut off labels
        
        plt.savefig(os.path.join(self.base_output_path, self.experiment_name, "images/histogram_per_token_{}.png".format(self.token_number)))
        
    
    def calculate_and_plot_sparsity(self, data):
        # Combine all dataframes into one
        combined_df = pd.concat(data)

        # Calculate sparsity for each layer
        # Considering a value sparse if it is exactly 0
        sparsity_df = combined_df.assign(is_sparse=lambda x: x['value'].abs() < 0.0001)
        layer_sparsity = sparsity_df.groupby('layer')['is_sparse'].mean()

        # Plotting the sparsity
        plt.figure(figsize=(12, 7))
        layer_sparsity.plot(kind='bar')
        plt.title('Overall Sparsity by Layer')
        plt.xlabel('Layer')
        plt.ylabel('Sparsity (%)')
        plt.ylim(0, 1)  # Ensure y-axis is from 0 to 1 to represent percentage
        plt.tight_layout()

        plt.savefig(os.path.join(self.base_output_path, self.experiment_name, "images/average_sparsity.png".format(self.token_number)))
               

    def sparcity_graph_per_token(self):
        # read csv
        for i in range(1, 88):
            df = pd.read_csv(f'/grphome/grp_inject/compute/logging/test6_config_boy_Llama-2-7b-hf_anger_QA_13b_2.pkl_0_1_2_3_4/Prompt1_CSVs/index_value_layer_{i}.csv')
            sparsity_df = df.groupby('layer')['value'].apply(self.calculate_sparsity).reset_index(name='sparsity')

            
            cmap = plt.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, len(sparsity_df['layer'])))

            # Plotting the histogram with the colormap
            plt.figure(figsize=(12, 8))
            plt.bar(sparsity_df['layer'], sparsity_df['sparsity'], color=colors)
            plt.xlabel('Layer')
            plt.ylabel('Sparsity (%)')
            plt.title('Histogram of Sparsity for Values < 0.01 by Layer')
            plt.xticks(rotation=45, ha="right")  # Rotate labels to avoid overlap
            plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
            
            plt.savefig(f'/grphome/grp_inject/compute/logging/test6_config_boy_Llama-2-7b-hf_anger_QA_13b_2.pkl_0_1_2_3_4/Sparsity_Plots/sparsity_{i}.png') 
            
               

    def calculate_sparsity(self, values):
        return (values < 0.01).mean() * 100
    
    
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

        
