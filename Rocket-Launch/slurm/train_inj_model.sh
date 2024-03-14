#!/bin/bash

#SBATCH --time=00:10:00   # walltime
#SBATCH --ntasks-per-node=2 # number of processor cores (i.e. tasks)
#SBATCH --nodes=2   # number of nodes
#SBATCH --mem=512G   # memory per CPU core
#SBATCH --gres=gpu:2
#SBATCH --qos=cs
#SBATCH -J "train_model"   # job name

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate rocket
srun python3 ../src/injected_train.py ../configs/test_config_anger_QA_13b_2.pkl_0_1_2_3.yaml