#!/bin/bash

#SBATCH --time=5:30:00   # walltime
#SBATCH --ntasks-per-node=8 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=512G   # memory per CPU core
#SBATCH --gpus=8
#SBATCH --qos=dw87
#SBATCH --partition=dw
##SBATCH --qos=cs
##SBATCH --partition=cs
#SBATCH -J "injected_train"   # job name
#SBATCH --output=%x_%j.out
#SBATCH --exclude=dw-1-4

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate test

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'

# srun python3 ../src/simple_injected_train.py ../configs/simple_injected_train.yaml
# srun python3 ../src/injected_train_tinyshakespeare.py /home/huang717/DRAGN/IRM/injectable-alignment-model/configs/Llama-2-7b-chat-hf_tiny_shakespeare_0_training.yaml
# srun python3 ../src/injected_train_tinyshakespeare.py /home/huang717/DRAGN/IRM/injectable-alignment-model/configs/Llama-2-7b-chat-hf_tiny_shakespeare_31_training.yaml
# srun python3 ../src/injected_train.py /home/huang717/DRAGN/IRM/injectable-alignment-model/configs/Llama-2-7b-chat-hf_anger_60k_15_training_new_test.yaml
srun python3 ../src/injected_train.py /home/huang717/DRAGN/IRM/injectable-alignment-model/configs/Llama-2-7b-chat-hf_sadness_60k_31_training_new_test.yaml
# srun python3 ../src/injected_train.py /home/huang717/DRAGN/IRM/injectable-alignment-model/configs/Llama-2-7b-chat-hf_neutral_60k_15_training_new_test.yaml