#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks-per-node=2 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=512G   # memory per CPU core
#SBATCH --gres=gpu:2
##SBATCH --qos=dw87
#SBATCH --qos=cs

#SBATCH -J "simple_injected_train"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate rocket

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'

# srun python3 ../src/simple_injected_train.py ../configs/simple_injected_train.yaml
srun python3 ../src/simple_injected_train.py ../configs/Llama-2-7b-chat-hf_anger_60k_0-31.yaml