#!/bin/bash

#SBATCH --time=02:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=16384M   # memory per CPU core
#SBATCH --gpus=1
#SBATCH --qos=cs
#SBATCH -J "tokenize"   # job name
#SBATCH --mail-user=drew.s.galbraith@gmail.com   # email address
#SBATCH --mail-type=END


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# nvidia-smi
mamba activate rocket_training
torchrun --nproc_per_node 1 ../llama_generation.py
# python3 llama_tokenizer.py -t