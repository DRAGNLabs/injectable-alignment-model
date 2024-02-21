import pandas as pd
import matplotlib as plt

from typing import List

def get_lengths(dataset: List[int], filter=True, min_length=0, max_length=2000, make_plot=True) -> pd.Series:
    """
    Get the lengths of sequences for a given list, intended to be used with Rocket_Dataset; Ex: get_lengths(dataset.train)

    filter: Bool, makes 
    """
    # range, mean, std dev?
    seq_lengths:pd.Series = dataset.apply(lambda x: len(x[1]), axis=1)

    # Filter the Series to include only sequences within the specified range
    if filter==True:
        seq_lengths = seq_lengths[(seq_lengths >= min_length) & (seq_lengths <= max_length)]
        
    if make_plot==False:
        return seq_lengths

    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(seq_lengths, bins=100, color='skyblue', edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Sequence Length Distribution (Histogram)')

    # Calculate the range
    data_range = seq_lengths.max() - seq_lengths.min()

    # Calculate the mean
    mean_value = round(seq_lengths.mean())

    # Calculate the median
    median_value = seq_lengths.median()
    
    # Calculate the standard deviation
    std_deviation = round(seq_lengths.std())

    # Calculate the Interquartile Range (IQR)
    Q1 = seq_lengths.quantile(0.25)
    Q3 = seq_lengths.quantile(0.75)
    IQR = Q3 - Q1

    # Define a threshold for identifying outliers (e.g., 1.5 times the IQR)
    outlier_threshold = 1.5 * IQR

    # Identify outliers
    outliers = seq_lengths[
        (seq_lengths < (Q1 - outlier_threshold)) | 
        (seq_lengths > (Q3 + outlier_threshold))
    ]

    # Print the results
    # print(f"Range: {data_range}")
    # print(f"Mean: {mean_value}")
    # print(f"Median: {median_value}")
    # print(f"Standard Deviation: {std_deviation}")
    # print("Outliers:")
    # print(outliers)
    plt.text(0.79, 0.9, f"Range: {data_range}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.79, 0.85, f"Mean: {mean_value}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.79, 0.8, f"Std Dev: {std_deviation}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.79, 0.75, f"Median: {median_value}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.79, 0.7, f"Q1: {Q1}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.79, 0.65, f"Q3: {Q3}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.79, 0.6, f"Outliers: {len(outliers)}", transform=plt.gca().transAxes, fontsize=12)

    # Add dotted lines at standard deviation intervals from the mean
    plt.axvline(x=mean_value, color='orange', linestyle='--', label='Mean')
    plt.axvline(x=mean_value - std_deviation, color='gray', linestyle='--', label='-1 Std Dev')
    plt.axvline(x=mean_value + std_deviation, color='gray', linestyle='--', label='+1 Std Dev')
    plt.axvline(x=mean_value - 2 * std_deviation, color='gray', linestyle='--', label='-2 Std Dev')
    plt.axvline(x=mean_value + 2 * std_deviation, color='gray', linestyle='--', label='+2 Std Dev')

    plt.savefig('gpt3-1_Sequence_Plot.png')

    return seq_lengths