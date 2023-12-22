#!/bin/bash

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks-per-node=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=32768M   # memory per CPU core
#SBATCH -J "single_train_IRM"   # job name


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate rocket
srun python3 ../injected_train_single.py ../configs/winter_train_single_config.yaml