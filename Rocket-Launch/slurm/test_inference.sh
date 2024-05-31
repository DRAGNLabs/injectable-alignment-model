#!/bin/bash --login

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH -J "inference_logging"   # job name
#SBATCH --qos=dw87
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate rocket

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

# srun python3 ../src/inference_logger.py ../configs/test_irm_run_thing.yaml
# srun python3 ../src/inference_logger.py ../configs/inference_logging/regularized_a=1e-4/config_Llama-2-7b-chat-hf_anger_QA_7b.pkl_0_1_2_3_4_5_6_7.yaml
# srun python3 ../src/inference_logger.py ../configs/inference_logging/non_regularized/config_Llama-2-7b-chat-hf_anger_QA_7b.pkl_0_1_2_3_4_5_6_7.yaml
# srun python3 ../src/inference_logger.py ../configs/inference_logging/regularized_a=1e-4/config_Llama-2-7b-chat-hf_anger_QA_7b.pkl_injected_layers_0-31.yaml
# srun python3 ../src/inference_logger.py ../configs/inference_logging/non_regularized/config_Llama-2-7b-chat-hf_anger_QA_7b.pkl_injected_layers_0-31.yaml
# srun python3 ../src/inference_logger.py ../configs/inference_logging/regularized_a=1e-4/config_Llama-2-7b-chat-hf_wikipedia.pkl_injected_layers_0-31_r.yaml

# srun python3 ../src/inference_logger.py ../configs/inference_logging/non_regularized/config_Llama-2-7b-chat-hf_wikipedia.pkl_injected_layers_0-31_r.yaml
# srun python3 ../src/inference_logger.py ../configs/inference_logging/non_regularized/config_Llama-2-7b-chat-hf_unpublished_books.pkl_injected_layers_0-31_r.yaml
# srun python3 ../src/inference_logger.py ../configs/inference_logging/regularized_a=1e-4/config_Llama-2-7b-chat-hf_unpublished_books.pkl_injected_layers_0-31_r.yaml
# srun python3 ../src/inference_logger.py ../configs/inference_logging/regularized_a=1e-4/config_Llama-2-7b-chat-hf_sadness_QA_7b.pkl_injected_layers_0-31_seed_420.yaml
# srun python3 ../src/inference_logger.py ../configs/inference_logging/regularized_a=1e-4/config_Llama-2-7b-chat-hf_anger_QA_7b.pkl_injected_layers_0-31_seed_420.yaml

srun python3 ../src/inference_logger.py ../configs/inference_logging/non_regularized/config_Llama-2-7b-chat-hf_neutral_QA_7b.pkl_24_25_26_27_28_29_30_31.yaml