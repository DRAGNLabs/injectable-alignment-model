#!/bin/bash

#SBATCH --time=01:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=64G
#SBATCH -J "single_train_IRM"   # job name


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate rocket
srun python3 ../injected_train_single.py ../configs/data_run_2_3_anger_output.yaml