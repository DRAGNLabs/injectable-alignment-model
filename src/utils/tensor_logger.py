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
    # TODO: Add logging functionality to inference loop
    # TODO: Clean up the logging code 

    def __init__(self, num_hidden_layers, experiment_name, layers, log_dir):
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
        self.sequence_len = 1

        self.prompt_df = pd.DataFrame()

        self.layered_tensor = torch.empty(0).to(self.device)
        self.all = torch.empty(0)
        self.means = torch.empty(0).to(self.device)
        self.std_dev = torch.empty(0).to(self.device)
        self.sparsity = torch.empty(0).to(self.device)
        self.max_activations = torch.empty(0).to(self.device)
        self.min_activations = torch.empty(0).to(self.device)
        self.modes = torch.empty(0).to(self.device)

        self.xticks = [i for i in range(4097) if i % 128 == 0]
        self.xticklabels = [i if i % 512 == 0 else "" for i in range(4097) if i % 128 == 0]
        self.yticks = [i for i in range(len(self.layers) + 1)]
        mod = len(self.layers) / 8
        self.yticklabels = [i if i % mod == 0 else "" for i in self.layers]
        self.yticklabels.append(self.layers[-1] + 1)

        self.base_output_path = f"{log_dir}/results" # "/grphome/grp_inject/compute/logging/new_prompts_runs_7_prompts_4"
        # self.base_output_path = "/grphome/grp_inject/compute/logging/testing"
        self.experiment_name = experiment_name  # config.experiment_name?
        os.makedirs(os.path.join(self.base_output_path, self.experiment_name), exist_ok=True)

    def new_prompt(self):
        self.token_number = 1
        self.prompt_number += 1
        self.sequence_len = 1
        
    def get_layer_weights(self, tensor, layer_id): return tensor[:, :, :, self.layers.index(layer_id)]
    
    def write_csv(self, df):
        if self.sequence_len <= self.token_number: name_of_csv = f'generated_token_{self.token_number - self.sequence_len + 1}.csv'
        else : name_of_csv = f'prompt_token_{self.token_number}.csv'
        self.output_path = os.path.join(self.base_output_path, self.experiment_name, "Prompt{}_CSVs".format(self.prompt_number))
        os.makedirs(self.output_path, exist_ok=True)
        df.to_csv(os.path.join(self.output_path,name_of_csv), index=False)
        
    def add_tensor(self, tensor: torch.Tensor):
        print(tensor.shape[1], flush=True)
        if tensor.shape[1] > 1:
            print("Splitting tensor", flush=True)
            self.sequence_len = tensor.shape[1]
            tensors = torch.split(tensor, 1, dim=1)
            for t in tensors: self.add_tensor(t)
        else:
            weights = torch.flatten(tensor, start_dim=0, end_dim=2).cpu().detach().numpy()
            
            print("Weights shape: {}\n\n".format(weights.shape), flush=True)
            print("Weights Length: {}\n\n".format(len(weights)), flush=True)
            dataFrames = []
            for i in range(weights.shape[1]):
                curr_weights = weights[:, i]
                curr_layer = self.layers[i]
                dataFrames.append(pd.DataFrame({'index': range(len(curr_weights)),  'layer': [curr_layer for j in range(len(curr_weights))], 'value': curr_weights}))
            self.token_df =  pd.concat(dataFrames)
            print(len(self.token_df))

            self.write_csv(self.token_df)
            self.token_number += 1

    def write_log(self):
        # Ensure the directory exists
        directory_path = os.path.join(self.base_output_path, self.experiment_name)
        os.makedirs(directory_path, exist_ok=True)

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

        output_path = os.path.join(self.base_output_path, self.experiment_name, 'images/', "prompt_{}".format(self.prompt_number - 1))
        os.makedirs(output_path, exist_ok=True)
        # Save the animation
        frames[0].save(os.path.join(output_path, "{0}_heatmap_animation{1}.gif".format(self.experiment_name, self.prompt_number - 1)), format='GIF', append_images=frames[1:], 
                    save_all=True, duration=200, loop=0)

    def generate_singel_heatmap(self, df, file, i, avg_max, avg_min):
        pivot_df = df.pivot(index="layer", columns="index", values="value")
        print("Pivoted dataframe:\n", flush = True)
        print(pivot_df.head(), flush = True)
        pivot_df_filled = pivot_df.fillna(0)
        heatmap_file_name = f'{file}_token_heatmap_{i}.png'
        print("\n\n\n MIN : {}\n\n\n".format(pivot_df.min().min()), flush = True)
        plt.figure(figsize=(14, 8))
        ax = sns.heatmap(pivot_df_filled, cmap="cividis", cbar_kws={'label': 'IRM Output Value'}, center = 0.0, vmin=avg_min if pivot_df.min().min() > avg_min else pivot_df.min().min(), vmax=avg_max if pivot_df.max().max() < avg_max else pivot_df.max().max())

        ax.set_xticks(self.xticks)
        ax.set_xticklabels(self.xticklabels)
        ax.set_yticks(self.yticks)
        ax.set_yticklabels(self.yticklabels)
        plt.show()
        plt.savefig(os.path.join(self.img_output_path, heatmap_file_name))
        plt.close()

    def recursively_generate_heatmap(self, path):
        data = dict()
        csv_identifiers = []
        csv_counter = 1
        if os.path.isdir(path):
            print("Path: ", path, flush=True)
            # Get and Sort files
            files = os.listdir(path)
            files = natsorted(files)

            for file in files:
                if os.path.isdir(os.path.join(path, file)):
                    self.recursively_generate_heatmap(os.path.join(path, file)) 
                else:
                    print("File: ", file, flush=True)
                    df = pd.read_csv(os.path.join(path, file))
                    df['csv_number'] = csv_counter
                    data[file] = df
                    csv_identifiers.append(csv_counter)
                    csv_counter += 1

        self.img_output_path = os.path.join(self.base_output_path, self.experiment_name, 'images/', "prompt_{}".format(self.prompt_number - 1))
        os.makedirs(self.img_output_path, exist_ok=True)

        self.generate_histograms(data.values())
        self.calculate_and_plot_sparsity(data.values())
        self.create_histogram_of_top_values_by_csv(data.values(), csv_identifiers)

        i = 1
        maxv = []
        minv = []
        for df in data.values():
            print("Dataframe{}:\n".format(i), flush = True)
            print(df.head(), flush = True)
            max_index = df['value'].idxmax()
            print("Max value and index: ", df.loc[max_index], flush = True)
            min_index = df['value'].idxmin()
            print("Min value and index: ", df.loc[min_index], flush = True)
            maxv.append(df['value'].max())
            minv.append(df['value'].min())
            i += 1
        self.max_value = max(maxv)
        print("Max value:{}".format(self.max_value), flush = True)
        self.min_value = min(minv)
        print("Min Value:{}".format(self.min_value), flush = True)

        avg_max = sum(maxv) / len(maxv)
        avg_min = sum(minv) / len(minv)

        vals = []
        i = 1
        name = "gen"
        lastname = "gen"
        for file in data.keys():
            name = file.split("_")[0]
            if name[0] != lastname[0]: i = 1
            self.generate_singel_heatmap(data.get(file), name, i, avg_max * 1.1, avg_min * 1.1)
            i += 1
            lastname = name

        print("Generating 3D heatmap", flush = True) 
        ### IF YOU WANT TO GENERATE A 3D HEATMAP, UNCOMMENT THE FOLLOWING LINE ###
        # self.generate_anim(data.values())

        vals = []
        for key in data.keys():
            if "generated" in key and len(vals) < 10: vals.append(data.get(key))
            if "prompt" in key: vals.insert(int(key.strip(".csv").split('_')[-1]) - 1, data.get(key))
        
        result = pd.concat(vals).groupby(['layer', 'index'])['value'].mean().reset_index()
        print("Result:\n", flush = True)
        print(result.head(), flush = True)
        pivot_df = result.pivot(index="layer", columns="index", values="value")
        print("Pivoted final dataframe:\n", flush = True)
        print(pivot_df.head(), flush = True)
        pivot_df = pivot_df.iloc[:self.num_hidden_layers]
        pivot_df_filled = pivot_df.fillna(0)

        heatmap_file_name = "Average_first_10_generated_token_value_heatmap.png"

        # Create the heatmap
        plt.figure(figsize=(14, 8))
        print("\n\n\n MIN : {}\n\n\n".format(pivot_df.min()), flush = True)
        # Use a colormap (cmap) that distinguishes your filled value (-1) from the rest, e.g., 'coolwarm'
        # You may need to adjust the colormap or the fill value depending on your data range and preferences
        ax = sns.heatmap(pivot_df_filled, cmap="cividis", cbar_kws={'label': 'Value'}, center = 0.0)
        ax.set_xticks(self.xticks)
        ax.set_xticklabels(self.xticklabels)
        ax.set_yticks(self.yticks)
        ax.set_yticklabels(self.yticklabels)

        plt.show()
        plt.savefig(os.path.join(self.img_output_path, heatmap_file_name))
        plt.close()

    def generate_heatmaps(self): self.recursively_generate_heatmap(self.output_path)


    def generate_histograms(self, data):
        combined_df = pd.concat(data)

        combined_df['abs_value'] = combined_df['value'].abs()
        top_2000 = combined_df.nlargest(100, 'abs_value')

        freq_distribution = top_2000['layer'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        freq_distribution.plot(kind='bar')
        plt.title('Frequency of Top 2000 Largest Magnitude IRM Weights by Layer')
        plt.xlabel('Layer')
        plt.ylabel('Frequency')
        
        os.makedirs(os.path.join(self.img_output_path, "histograms"), exist_ok=True)
        plt.savefig(os.path.join(self.img_output_path, "histograms/histogram_per_layer.png"))
        plt.close()
        
    def generate_average_histograms(self, data):
        grouped_data = [data[i:i + 4] for i in range(0, len(data), 4)]
        for group in grouped_data:
            combined_df = pd.concat(group).groupby(['layer', 'index'], as_index=False).mean()
            pivot_df = combined_df.pivot(index="layer", columns="index", values="value").fillna(0)

            plt.figure(figsize=(14, 8))
            sns.heatmap(pivot_df, cmap="cividis", cbar_kws={'label': 'Value'}, center = 0.0)
            plt.ylabel('Layer')
            plt.xlabel('Index')
            plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

            heatmap_file_name = f'average_heatmap_group_{self.prompt_number - 1}.png'
            output_path = os.path.join(self.base_output_path, self.experiment_name, 'images', f"prompt_{self.prompt_number - 1}")
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, heatmap_file_name))
            plt.close()
        
    def get_first_word(self, experiment_name): return experiment_name.split("_")[0]

    def create_histogram_of_top_values_by_csv(self, data, csv_identifiers):
        combined_df = pd.concat(data)

        combined_df['abs_value'] = combined_df['value'].abs()

        top_2000 = combined_df.nlargest(1000, 'abs_value')
        freq_distribution_by_csv = top_2000['csv_number'].value_counts().reindex(csv_identifiers, fill_value=0)

        # Plotting the histogram
        plt.figure(figsize=(12, 7))
        freq_distribution_by_csv.plot(kind='bar')
        plt.title('Frequency of Top 2000 Largest Magnitude {} IRM Outputs by Token'.format(self.get_first_word(self.experiment_name)))
        plt.xlabel('Token Index')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to not cut off labels

        os.makedirs(os.path.join(self.base_output_path, self.experiment_name, "images/"), exist_ok=True)
        
        plt.savefig(os.path.join(self.base_output_path, self.experiment_name, "images/histogram_per_token_{}.png".format(self.prompt_number - 1)))
        plt.close()

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
        plt.close()

    def calculate_sparsity(self, values): return (values < 0.01).mean() * 100
