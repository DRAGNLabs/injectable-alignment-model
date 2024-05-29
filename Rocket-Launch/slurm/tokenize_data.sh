#!/bin/bash --login

#SBATCH --time=00:15:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=256G   # memory per CPU core
#SBATCH --qos=dw87
#SBATCH -J "data_tokenize"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate rocket

# python3 ../src/tokenize_data.py ../configs/experiment_configs/config_Llama-2-7b-chat-hf_sadness_QA_7b.pkl_injected_layers_0-31.yaml
# python3 ../src/tokenize_data.py ../configs/config_Llama-2-7b-chat-hf_unpublished_books.pkl_injected_layers_0-31_r.yaml
python3 ../src/tokenize_data.py ../configs/config_Llama-2-7b-chat-hf_wikipedia.pkl_injected_layers_0-31_r.yaml
