#!/bin/bash

#SBATCH --time=03:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=256G   # memory per CPU core
#SBATCH -J "train_tokenizer"   # job name

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate rocket

python3 ../tokenizer/train_tokenizer.py "--input=../dataset/raw/openorca_combined.csv --input_format=text --input_sentence_size=1000000 --train_extremely_large_corpus=true --model_prefix=tokenizer.large --vocab_size=32000 --shuffle_input_sentence=true --pad_id=3"