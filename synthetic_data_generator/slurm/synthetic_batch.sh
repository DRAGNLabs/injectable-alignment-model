#!/bin/bash

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1 -C kepler
#SBATCH --mem-per-cpu=4096M   # memory per CPU core

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi

mamba activate irm
python -u ../synthetic_data_generator/run_data.py $1

