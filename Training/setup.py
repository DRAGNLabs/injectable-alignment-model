import subprocess
from transformers import AutoTokenizer

def install_requirements():
    requirements_file = "requirements.txt" 
    print(f'Installing requirements...')
    try:
        subprocess.check_call(['pip', 'install', '-r', requirements_file])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")

def download_llama():
    print(f'Downloading Llama')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def download_dataset():
    print(f'Downloading dataset')
    gpt4 = [
        "curl",
        "-L",
        "-o",
        "1M-GPT4-Augmented.parquet",
        "https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/1M-GPT4-Augmented.parquet"
    ]
    gpt3_5 = [
        "curl",
        "-L",
        "-o",
        "3_5M-GPT3_5-Augmented.parquet",
        "https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/3_5M-GPT3_5-Augmented.parquet"
    ]

    try:
        subprocess.check_call(gpt4)
        subprocess.check_call(gpt3_5)
        print("File downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")

def tokenize_dataset():
    print(f'Tokenizing dataset. Check latest slurm.out file for time left.')
    sbatch_command = [
        "sbatch",
        "job.sh"
    ]

    try:
        subprocess.check_call(sbatch_command)
        print("Job submitted successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")

if __name__ == "__main__":
    install_requirements()
    download_llama()
    download_dataset()
    tokenize_dataset()