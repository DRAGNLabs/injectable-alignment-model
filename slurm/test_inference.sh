#!/bin/bash

#SBATCH --time=1:30:00   # walltime
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --gpus=1
#SBATCH -J "inference_logging"   # job name
#SBATCH --qos=dw87
#SBATCH --partition=dw
## SBATCH --qos=cs
## SBATCH --partition=cs
#SBATCH --output=%x_%j.out
#SBATCH --exclude=dw-1-4

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate test

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'

# srun python3 ../src/injected_inference.py /home/huang717/DRAGN/IRM/injectable-alignment-model/configs/Llama-2-7b-chat-hf_tiny_shakespeare_31_inference.yaml
# srun python3 ../src/injected_inference.py /home/huang717/DRAGN/IRM/injectable-alignment-model/configs/Llama-2-7b-chat-hf_anger_60k_31_inference_latest.yaml
# srun python3 ../src/injected_inference.py /home/huang717/DRAGN/IRM/injectable-alignment-model/configs/Llama-2-7b-chat-hf_sadness_60k_31_inference_latest.yaml
srun python3 ../src/injected_inference.py /home/huang717/DRAGN/IRM/injectable-alignment-model/configs/Llama-2-7b-chat-hf_neutral_60k_31_inference_latest.yaml