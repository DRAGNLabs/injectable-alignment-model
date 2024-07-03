#!/bin/bash --login

#SBATCH --time=03:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=256G   # memory per CPU core
#SBATCH -J "train_sp_tokenizer"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate rocket

python3 ../src/train_sp_tokenizer.py \
        "--input=/PATH_TO_DATA/dataset/raw/PATH_TO_DATA.parquet --input_format=text --input_sentence_size=1000000 --train_extremely_large_corpus=false --model_prefix=tokenizer --vocab_size=32000 --shuffle_input_sentence=true --pad_id=3" \
        ../configs/PATH_TO_CONFIG.yaml
        