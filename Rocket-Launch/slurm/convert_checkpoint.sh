#!/bin/bash

#SBATCH --time=00:15:00   # walltime
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --nodes=2   # number of nodes
#SBATCH --mem=512G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --qos=cs
#SBATCH -J "train_model"   # job name

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate rocket
srun python3 ../src/convert_checkpoint.py ../configs/test_config_13b_anger_QA_13b_2.pkl_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39.yaml