import os
import subprocess
from transformers import AutoTokenizer

os.system('huggingface-cli login --token $HF_TOKEN')  # Ensure you have either [A] exported an env var w/ the corresponding token (recommended) or [B] replaced '$HF_TOKEN' w/ your exact token

def download_llama():
    print(f'\n\nDownloading Llama\r')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    print("\033[1;32m Downloaded Llama-2! \033[1;30m\n\n")

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
    print('\n\n')
    sbatch_command = [
        "sbatch",
        "tokenize_job.sh"
    ]

    try:
        subprocess.check_call(sbatch_command)
        print("Job submitted successfully!")
        print(f'Tokenizing dataset.\n Check latest slurm.out file for estimated time left.')
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")

if __name__ == "__main__":
    download_llama()
    download_dataset()
    tokenize_dataset()
