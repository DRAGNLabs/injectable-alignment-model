# Setup process
2. Request access to Llama 2 https://ai.meta.com/llama/ using the same email address as your Hugging Face account
3. Request access to 'Llama-2-7b-chat-hf' through Hugging Face. You must first request a download from the Meta AI website using the same email address as your Hugging Face account. After doing so, you can request access to any of the models on Hugging Face and within 1-2 days your account will be granted access to all versions.
1. Run "mamba create --name rocket_training python=3.11"
1. Activate the environment to run scripts with "mamba activate rocket_training"
1. Run "pip install -r requirements.txt"
3. Run "python setup.py" to complete setup

# Train
Use train_job.sh for training with a single gpu. Submit the job by using "sbatch job.sh"

    2. pip install -r requirements.txt
    1. Make a folder called "dataset" that is on the same level as this readme.md
    1. Run these commands from inside the "dataset" folder:
        1. curl -L -o 1M-GPT4-Augmented.parquet https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/1M-GPT4-Augmented.parquet
        1. curl -L -o 3_5M-GPT3_5-Augmented.parquet https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/3_5M-GPT3_5-Augmented.parquet
    1. Change "test_gpu" in each .sh file to the name of your conda environment
    4. The .sh files are made to run on slurm with "sbatch job.sh" for example
    1. Request access to Llama 2 https://ai.meta.com/llama/
    5. First run tokenize_job.sh to tokenize the dataset. This takes about 6 hours
    6. Then run job.sh to train. Models will save to the models folder that is at the same level as this "Training" directory.
    7. Checkpoints will appear in an "checkpoints" directory in the same level as this readme.md