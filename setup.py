import os
import subprocess
import pandas as pd

def download_dataset():
    print(f'\n\nDownloading dataset')
    directory_name = "dataset/raw"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    gpt4_filename = f"{directory_name}/1M-GPT4-Augmented.parquet"
    gpt3_5_filename = f"{directory_name}/3_5M-GPT3_5-Augmented.parquet"

    if not os.path.exists(gpt4_filename):
        try:
            subprocess.check_call([
                "curl",
                "-L",
                "-o",
                gpt4_filename,
                "https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/1M-GPT4-Augmented.parquet"
            ])
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e}")
            exit()

    if not os.path.exists(gpt3_5_filename):
        try:
            subprocess.check_call([
                "curl",
                "-L",
                "-o",
                gpt3_5_filename,
                "https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/3_5M-GPT3_5-Augmented.parquet"
            ])
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e}")
            exit()

def combine_parquet_files():
    print(f'\n\nCombining parquet files')
    directory_name = "dataset/raw"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    gpt4_filename = f"{directory_name}/1M-GPT4-Augmented.parquet"
    gpt3_5_filename = f"{directory_name}/3_5M-GPT3_5-Augmented.parquet"
    combined_parquet_files = f"{directory_name}/openorca_combined.parquet"

    # Read in the two parquet files
    gpt4_df = pd.read_parquet(gpt4_filename)
    gpt3_5_df = pd.read_parquet(gpt3_5_filename)

    # Combine the two dataframes
    combined_df = pd.concat([gpt4_df, gpt3_5_df])

    # Write the combined dataframe to a new parquet file
    combined_df.to_parquet(combined_parquet_files)

def generate_raw_csv_file():
    # We need to pass a raw text file to sentencepiece to train a tokenizer
    # Additionally, we only want to train the tokenizer on the raw data, not the id
    print(f'\n\nGenerating raw csv file')
    directory_name = "dataset/raw"

    combined_parquet_files = f"{directory_name}/openorca_combined.parquet"
    raw_csv_file = f"{directory_name}/openorca_combined.csv"

    # Read in the parquet file
    combined_df = pd.read_parquet(combined_parquet_files)

    # Drop the id column
    combined_df = combined_df.drop(columns=['id'], axis=1)

    # Write the combined dataframe to a new csv file
    combined_df.to_csv(raw_csv_file, index=False)

def main():
    download_dataset()
    combine_parquet_files()
    generate_raw_csv_file()

if __name__ == "__main__":
    main()