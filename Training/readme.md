Here's the setup process:
    1. Make a virtual environment for training
    2. pip install -r requirements.txt
    3. Download the two .parquet files from https://huggingface.co/datasets/Open-Orca/OpenOrca/tree/main and put them into a folder called "dataset" that is on the same level as this readme.md
    4. The .sh files are made to run on slurm with "sbatch job.sh" for example
    5. First run tokenize_job.sh to tokenize the dataset. This takes about 6 hours
    6. Then run job.sh to train. Models will save to the models folder that is at the same level as this "Training" directory.
    7. Checkpoints will appear in an "checkpoints" directory in the same level as this readme.md