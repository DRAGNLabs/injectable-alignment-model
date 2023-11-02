#!/bin/bash

#SBATCH --time=00:20:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=16384M   # memory per CPU core
#SBATCH --gpus=1
#SBATCH --qos=cs
#SBATCH -J "test_gpu"   # job name

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate rocket
#torchrun --nproc_per_node 1 ../llama_generation.py
srun python3 ../run.py ../configs/train_config.yaml