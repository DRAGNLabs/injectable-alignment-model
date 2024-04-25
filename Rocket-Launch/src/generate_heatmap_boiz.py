import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors 

csv_files = [
    "/home/myl15/fsl_groups/grp_inject/compute/logging/untrained_irm/Prompt3_CSVs/index_value_layer_1.csv", 
    "/home/myl15/fsl_groups/grp_inject/compute/logging/anger_test_0-7/Prompt1_CSVs/index_value_layer_14.csv", 
    "/home/myl15/fsl_groups/grp_inject/compute/logging/sadness_test_0-7/Prompt3_CSVs/index_value_layer_17.csv", 
    "/home/myl15/fsl_groups/grp_inject/compute/logging/neutral_test_0-7/Prompt1_CSVs/index_value_layer_1.csv"
]

# Create a dictionary to store DataFrames
dataframes = {}
names = ["Untrained", "Anger", "Sadness", "Neutral"]
nums = [1, 14, 17, 1]

# Load CSV files into DataFrames
for i in range(len(csv_files)):
    name = names[i]
    dataframes[name] = pd.read_csv(csv_files[i])

def min_max_from_dataframes(dataframes):
    """Calculates the overall minimum and maximum across all DataFrames, based on the 'value' column."""
    all_values = pd.concat([df['value'] for df in dataframes.values()])
    return all_values.min(), all_values.max()

overall_min, overall_max = min_max_from_dataframes(dataframes)

# Configure figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Generate Heatmaps
i = 0
for name, df in dataframes.items():
    pivot_df = df.pivot(index="layer", columns="index", values="value")
    pivot_df_filled = pivot_df.fillna(0)

    # Determine subplot position based on counter 'i'
    row = i // 2
    col = i % 2

    # Generate heatmap with common color scale
    sns.heatmap(
        pivot_df_filled, 
        cmap="coolwarm", 
        cbar_kws={'label': 'Value'}, 
        vmin=pivot_df.min().min(),  # Consistent scale across heatmaps
        vmax=pivot_df.max().max(),   # Consistent scale across heatmaps
        ax=axes[row, col] 
    )

    axes[row, col].set_title(f'Heatmap for {name} IRM at Step {nums[i]}')
    axes[row, col].set_ylabel('Layer')
    axes[row, col].set_xlabel('Index')

    i += 1

# Overall title and layout adjustments
fig.suptitle('Heatmaps of IRM Outputs for Different Models', fontsize=14)
plt.tight_layout()

# Saving 
output_path = "/home/myl15/inject/injectable-alignment-model/Rocket-Launch/src"
plt.savefig(os.path.join(output_path, 'combined_heatmaps.png'))
plt.close()