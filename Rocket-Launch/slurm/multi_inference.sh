#!/bin/bash --login

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --qos=dw87
#SBATCH -J "inference_logging"   # job name
#SBATCH --output=%x_%j.out

#SBATCH --array=15-17 #Number of jobs in the inference_confic.txt file. DONT FORGET TO UPDATE!


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE


# Specify the path to the config file
# config=/home/dfbaker5/cs301r/irm_sanbox/injectable-alignment-model-hugging-face/Rocket-Launch/slurm/inference_paths.txt
# config=inference_paths.txt
config=inference_paths_little.txt

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
config_path=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate rocket

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

echo "Running inference for $config_path"
srun python3 ../src/inference_logger.py $config_path