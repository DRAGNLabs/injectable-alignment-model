import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from natsort import natsorted

def generate_average_generated_token_heatmap(directory_path, output_path=None, save=False):
    """
    Generates a heatmap showing the average activation values across 
    the first 10 generated tokens only.
    
    This function is completely standalone and can be run independently
    of the tensor_logger.py file.
    
    Args:
        directory_path (str): Directory path containing CSV files with IRM activation data
        output_path (str, optional): Directory to save the output heatmap. 
                                    If None, uses directory_path.
    
    Returns:
        pandas.DataFrame: The pivoted dataframe used to generate the heatmap
    """
    # Set output path to input path if not specified
    if output_path is None:
        output_path = directory_path
    
    # Find all generated token CSV files
    generated_files = []
    
    # Check if path is a directory
    if os.path.isdir(directory_path):
        # Get all files and sort them naturally
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        files = natsorted(files)
        
        # Identify generated token files
        for file in files:
            if "generated" in file:
                generated_files.append(os.path.join(directory_path, file))
    
    # Take only the first N generated files
    n = 20
    generated_files = generated_files[:n]
    
    if not generated_files:
        print("No generated token files found in directory:", directory_path)
        return None
    
    print(f"Using {len(generated_files)} generated token files for average heatmap.")
    
    # Load and process the files
    dataframes = []
    for file in generated_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"Loaded file: {os.path.basename(file)}")
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    
    if not dataframes:
        print("No valid CSV files could be loaded.")
        return None
    
    # Combine all DataFrames and calculate average by layer and index
    combined_df = pd.concat(dataframes)
    average_df = combined_df.groupby(['layer', 'index'])['value'].mean().reset_index()
    
    # Pivot data for heatmap format
    pivot_df = average_df.pivot(index="layer", columns="index", values="value")
    
    # Fill NaN values with zeros
    pivot_df_filled = pivot_df.fillna(0)
    
    # Calculate min/max for color scaling
    min_val = pivot_df_filled.min().min()
    max_val = pivot_df_filled.max().max()
    
    # Generate the heatmap visualization
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(
        pivot_df_filled, 
        cmap="cividis", 
        cbar_kws={'label': 'Average Activation Value'},
        center=0.0,
        vmin=min_val,
        vmax=max_val
    )
    
    # Set up custom tick marks for better readability
    # Create tick marks at regular intervals
    max_index = pivot_df_filled.columns.max() if len(pivot_df_filled.columns) > 0 else 0
    xticks = [i for i in range(0, max_index + 1, 128) if i <= max_index]
    xticklabels = [i if i % 512 == 0 else "" for i in xticks]
    
    # Set ticks for layers
    layers = sorted(pivot_df_filled.index.unique())
    yticks = np.arange(len(layers))
    yticklabels = layers
    
    # Apply ticks
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    
    # Add title and labels
    plt.title(f"Average Activation Values (First {n} Generated Tokens Only)")
    plt.xlabel("Token Position Index")
    plt.ylabel("Layer")
    
    if save:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        # Save the heatmap
        heatmap_file_name = f"average_first_{n}_generated_tokens_heatmap.png"
        output_file_path = os.path.join(output_path, heatmap_file_name)
        plt.savefig(output_file_path)
        print(f"Generated heatmap saved as: {output_file_path}")
    plt.close()
    
    return average_df, pivot_df_filled  # Return the data for further analysis if needed

def plot_heatmap(df, title=None):
    # Pivot data for heatmap format
    pivot_df = df.pivot(index="layer", columns="index", values="value")

    # Fill NaN values with zeros
    pivot_df_filled = pivot_df.fillna(0)

    # Calculate min/max for color scaling
    min_val = pivot_df_filled.min().min()
    max_val = pivot_df_filled.max().max()
    
    # Find the max absolute value for symmetric color scaling
    abs_max = max(abs(min_val), abs(max_val))
    
    # Create a custom diverging palette (blue to white to red)
    # 220 is blue, 20 is red in hue values
    custom_cmap = sns.diverging_palette(220, 20, as_cmap=True)
    # custom_cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)

    ticks = np.linspace(-abs_max, abs_max, 7)

    # Generate the heatmap visualization
    plt.figure(figsize=(100, 8))
    ax = plt.gca()
    sns.heatmap(
        pivot_df_filled, 
        cmap='coolwarm',  # Use the custom diverging colormap
        cbar_kws={'label': 'Average Activation Value',
            'ticks': ticks,  # Specify the ticks you want to show
            'format': '%.2f'  # Format the tick labels
            },
        center=0.0,
        vmin=-abs_max,
        vmax=abs_max,
        # vmin=min_val,
        # vmax=max_val,
        ax = ax
    )
    # ax.set_aspect(0.1)

    # Set up custom tick marks for better readability
    max_index = pivot_df_filled.columns.max() if len(pivot_df_filled.columns) > 0 else 0
    xticks = [i for i in range(0, max_index + 1, 128) if i <= max_index]
    xticklabels = [i if i % 128 == 0 else "" for i in xticks]

    # Set ticks for layers
    layers = sorted(pivot_df_filled.index.unique())
    yticks = np.arange(len(layers))
    yticklabels = layers

    # Apply ticks
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Add title and labels
    if title:
        plt.title(title)
    plt.xlabel("Token Position Index")
    plt.ylabel("Layer")
    plt.show()

def subtract_df2_from_df1(df1, df2):
    df2_dict = dict(zip(zip(df2['layer'], df2['index']), df2['value']))
    df1_minus_df2 = df1.copy()
    df1_minus_df2['value'] = df1.apply(
        lambda row: row['value'] - df2_dict.get((row['layer'], row['index']), 0), 
        axis=1
    )
    return df1_minus_df2

if __name__ == "__main__":

    for prompt_idx in range(11):
        prompt_idx += 1
        angry_path = f"/home/huang717/DRAGN/IRM/injectable-alignment-model/runs/with_regularization/Anger60k_L1_on_outputs_irm_layer_31/results/Llama-2-7b-chat-hf_anger_60k_31_inference/Prompt{prompt_idx}_CSVs"
        sadness_path = f"/home/huang717/DRAGN/IRM/injectable-alignment-model/runs/with_regularization/Sadness60k_L1_on_outputs_irm_layer_31/results/Llama-2-7b-chat-hf_sadness_60k_31_inference/Prompt{prompt_idx}_CSVs"
        neutral_path = f"/home/huang717/DRAGN/IRM/injectable-alignment-model/runs/with_regularization/Neutral60k_L1_on_outputs_irm_layer_31/results/Llama-2-7b-chat-hf_neutral_60k_31_inference/Prompt{prompt_idx}_CSVs"
        mean_output_file_path = f"/home/huang717/DRAGN/IRM/injectable-alignment-model/src/utils"
        angry_output_path = '/home/huang717/DRAGN/IRM/injectable-alignment-model/runs/with_regularization/Anger60k_L1_on_outputs_irm_layer_31/results/Llama-2-7b-chat-hf_anger_60k_31_inference/'
        sadness_output_path = '/home/huang717/DRAGN/IRM/injectable-alignment-model/runs/with_regularization/Sadness60k_L1_on_outputs_irm_layer_31/results/Llama-2-7b-chat-hf_sadness_60k_31_inference/'
        neutral_output_path = '/home/huang717/DRAGN/IRM/injectable-alignment-model/runs/with_regularization/Neutral60k_L1_on_outputs_irm_layer_31/results/Llama-2-7b-chat-hf_neutral_60k_31_inference/'

        angry_avg_csv = generate_average_generated_token_heatmap(angry_path, output_file_path)[0]
        sadness_avg_csv = generate_average_generated_token_heatmap(sadness_path, output_file_path)[0]
        neutral_avg_csv = generate_average_generated_token_heatmap(neutral_path, output_file_path)[0]

        combined_df = pd.concat([angry_avg_csv, sadness_avg_csv, neutral_avg_csv])
        mean_of_3 = combined_df.groupby(['layer', 'index'])['value'].mean().reset_index()

        angry_minus_mean = subtract_df2_from_df1(angry_avg_csv, mean_of_3)
        sadness_minus_mean = subtract_df2_from_df1(neutral_avg_csv, mean_of_3)
        neutral_minus_mean = subtract_df2_from_df1(sadness_avg_csv, mean_of_3)



    pass