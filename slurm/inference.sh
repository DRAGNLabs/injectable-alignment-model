#!/bin/bash --login

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=256G   # memory per CPU core
#SBATCH --gpus=1
#SBATCH -J "inference"   # job name
#SBATCH --requeue
#SBATCH --signal=SIGHUP@90
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate rocket
srun python3 ../src/inference.py ../configs/PATH_TO_CONFIG.yaml
