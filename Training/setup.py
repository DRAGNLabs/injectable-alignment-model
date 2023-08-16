import os
import subprocess
from transformers import AutoTokenizer

def download_llama():
    print(f'\n\nDownloading Llama')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def download_dataset():
    print(f'\n\nDownloading dataset')
    directory_name = "dataset"
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

    print("Files downloaded successfully!")

def tokenize_dataset():
    print(f'\n\nTokenizing dataset. Check latest slurm.out file for time left.')
    sbatch_command = [
        "sbatch",
        "tokenize_job.sh"
    ]

    try:
        subprocess.check_call(sbatch_command)
        print("Job submitted successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")

if __name__ == "__main__":
    download_llama()
    download_dataset()
    tokenize_dataset()
