#!/bin/bash --login

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH -J "inference_logging"   # job name
##SBATCH --qos=dw87
#SBATCH --qos=cs
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate rocket

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'

srun python3 ../src/inference_logger.py ../configs/simple_injected_train.yaml