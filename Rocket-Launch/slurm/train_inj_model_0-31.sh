#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks-per-node=2 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=512G   # memory per CPU core
#SBATCH --gres=gpu:2
#SBATCH --qos=dw87

##SBATCH -J "0-31_r_anger"   # job name
##SBATCH -J "0-31_nr_anger"   # job name
##SBATCH -J "0-31_r_neutral"   # job name
##SBATCH -J "0-31_nr_neutral"   # job name
##SBATCH -J "0-31_r_sadness"   # job name
##SBATCH -J "0-31_nr_sadness"   # job name
#SBATCH -J "0-31_r_unpublished_books"   # job name
##SBATCH -J "0-31_r_wikipedia"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate rocket

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'

# srun python3 ../src/injected_train.py ../configs/regularized_configs/config_Llama-2-7b-chat-hf_anger_QA_7b.pkl_injected_layers_0-31.yaml
# srun python3 ../src/injected_train.py ../configs/none_regularized_configs/config_Llama-2-7b-chat-hf_anger_QA_7b.pkl_injected_layers_0-31.yaml
# srun python3 ../src/injected_train.py ../configs/regularized_configs/config_Llama-2-7b-chat-hf_neutral_QA_7b.pkl_injected_layers_0-31.yaml
# srun python3 ../src/injected_train.py ../configs/none_regularized_configs/config_Llama-2-7b-chat-hf_neutral_QA_7b.pkl_injected_layers_0-31.yaml
# srun python3 ../src/injected_train.py ../configs/regularized_configs/config_Llama-2-7b-chat-hf_sadness_QA_7b.pkl_injected_layers_0-31.yaml
# srun python3 ../src/injected_train.py ../configs/none_regularized_configs/config_Llama-2-7b-chat-hf_sadness_QA_7b.pkl_injected_layers_0-31.yaml

# srun python3 ../src/injected_train.py ../configs/config_Llama-2-7b-chat-hf_anger_QA_7b.pkl_injected_layers_0-31_r.yaml
srun python3 ../src/injected_train.py ../configs/config_Llama-2-7b-chat-hf_unpublished_books.pkl_injected_layers_0-31_r.yaml
# srun python3 ../src/injected_train.py ../configs/config_Llama-2-7b-chat-hf_wikipedia.pkl_injected_layers_0-31_r.yaml