#!/bin/bash --login

#SBATCH --time=3-00:00:00   # walltime (3 days, the maximum)
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=256000M   # RAM per CPU core
#SBATCH -J "experiment_inference"   # job name
#SBATCH --gpus=1
#SBATCH --qos=dw87
#SBATCH --requeue
#SBATCH --signal=SIGHUP@90
#SBATCH --output=%x_%j.out
#SBATCH --array=1-3 #Number of jobs in the inference_confic.txt file. DONT FORGET TO UPDATE!


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE


# Specify the path to the config file
config=/home/myl15/inject/injectable-alignment-model/Rocket-Launch/src/new_inference.txt

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
config_path=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate rocket

nvidia-smi
echo "Running inference for $config_path"
srun python3 c_sandbox.py $config_path