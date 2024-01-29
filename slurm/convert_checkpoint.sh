#!/bin/bash

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32G   # memory per CPU core
#SBATCH --gpus=1
#SBATCH -J "tokenize_data"   # job name

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate rocket

srun python3 /grphome/grp_inject/injectable-alignment-model/convert_checkpoint.py /grphome/grp_inject/injectable-alignment-model/configs/convert_checkpoint_config.yaml /grphome/grp_inject/injectable-alignment-model/injected_model_weights_6_7.ckpt