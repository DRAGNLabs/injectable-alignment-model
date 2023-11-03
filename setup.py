import os
import subprocess

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

download_dataset()