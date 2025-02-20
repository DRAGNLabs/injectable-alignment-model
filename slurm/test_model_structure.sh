#!/bin/bash

#SBATCH --time=0:20:00   # walltime
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=256G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --qos=cs
#SBATCH --partition=cs

#SBATCH -J "testing_model_structure"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate test

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'

# srun python3 ../src/simple_injected_train.py ../configs/simple_injected_train.yaml
srun python3 ../src/test_model_structure.py $1