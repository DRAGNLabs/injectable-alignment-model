#!/bin/bash

# sbatch --mail-user huang717@byu.edu --job-name "Test training on chat llama3 branch on neutral60k with IRM layers 31 with context_window=256 lr=1e-4 epoch=15 irm_size=1024x4" injected_train.sh
# sbatch --mail-user huang717@byu.edu --job-name "Test inference on chat llama3 branch on tiny_shakespeare with IRM layers 31 with context_window=256 lr=1e-5 epoch=10 irm_size=1024x4" test_inference.sh
# sbatch --mail-user huang717@byu.edu --job-name "Test inference on chat llama3 branch on anger60k with IRM layers 31 with context_window=256 lr=1e-4 epoch=15 irm_size=1024x4" test_inference.sh
# sbatch --mail-user huang717@byu.edu --job-name "Test inference on chat llama3 branch on sadness60k with IRM layers 31 with context_window=256 lr=1e-4 epoch=15 irm_size=1024x4" test_inference.sh
sbatch --mail-user huang717@byu.edu --job-name "Test inference on chat llama3 branch on neutral60k with IRM layers 31 with context_window=256 lr=1e-4 epoch=15 irm_size=1024x4" test_inference.sh