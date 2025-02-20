#!/bin/bash

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'

srun python3 ../src/injected_train.py ../configs/Llama-2-7b-chat-hf_anger_60k_31_training_new_test.yaml
