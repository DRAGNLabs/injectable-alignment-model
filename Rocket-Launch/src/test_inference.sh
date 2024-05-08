#!/bin/bash --login

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=128G   # memory per CPU core
#SBATCH --gpus=1
#SBATCH -J "inference_logging"   # job name
#SBATCH --qos=cs
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate rocket

# srun python3 inference_logger.py ../configs/test_irm_run_thing.yaml

srun python3 inference_logger.py ../configs/inference_logging/regularized_a=1e-4/config_Llama-2-7b-chat-hf_anger_QA_7b.pkl_0_1_2_3_4_5_6_7.yaml

