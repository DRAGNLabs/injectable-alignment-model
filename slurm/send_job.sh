#!/bin/bash

# sbatch --mail-user huang717@byu.edu --job-name "Test training on chat llama3 branch on anger60k with IRM layers 31 with context_window=256 lr=1e-4 epoch=15 irm_size=1024x4 L2 weight decay" injected_train_anger60k.sh
# sbatch --mail-user huang717@byu.edu --job-name "Test training on chat llama3 branch on sadness60k with IRM layers 31 with context_window=256 lr=1e-4 epoch=15 irm_size=1024x4 L2 weight decay" injected_train_sadness60k.sh
# sbatch --mail-user huang717@byu.edu --job-name "Test training on chat llama3 branch on neutral60k with IRM layers 31 with context_window=256 lr=1e-4 epoch=15 irm_size=1024x4 L2 weight decay" injected_train_neutral60k.sh
# sbatch --mail-user huang717@byu.edu --job-name "Training on chat llama3 branch on tiny_shakespeare with IRM layers 31 with context_window=4096 lr=5e-5 epoch=5 irm_size=1024x4 L2 weight decay" injected_train_tinyshakespeare.sh
# sbatch --mail-user huang717@byu.edu --job-name "Test inference on chat llama3 branch on tiny_shakespeare with IRM layers 31 with context_window=4096 lr=5e-5 epoch=5 irm_size=1024x4 L2 weight decay" test_inference.sh
# sbatch --mail-user huang717@byu.edu --job-name "Test inference on chat llama3 branch on anger60k with IRM layers 31 with context_window=256 lr=1e-4 epoch=15 irm_size=1024x4" test_inference.sh
# sbatch --mail-user huang717@byu.edu --job-name "Test inference on chat llama3 branch on sadness60k with IRM layers 31 with context_window=256 lr=1e-4 epoch=15 irm_size=1024x4" test_inference.sh
# sbatch --mail-user huang717@byu.edu --job-name "Test inference on chat llama3 branch on neutral60k with IRM layers 31 with context_window=256 lr=1e-4 epoch=15 irm_size=1024x4" test_inference.sh

# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 31, training" with_regularization/injected_train_anger60k_L1_on_outputs_irm_layer_31.sh
# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 31, training" with_regularization/injected_train_sadness60k_L1_on_outputs_irm_layer_31.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 31, training" with_regularization/injected_train_neutral60k_L1_on_outputs_irm_layer_31.sh

# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 0, training" with_regularization/injected_train_anger60k_L1_on_outputs_irm_layer_0.sh
# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 0, training" with_regularization/injected_train_sadness60k_L1_on_outputs_irm_layer_0.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 0, training" with_regularization/injected_train_neutral60k_L1_on_outputs_irm_layer_0.sh

# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 15, training" with_regularization/injected_train_anger60k_L1_on_outputs_irm_layer_15.sh

# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 31, inference" with_regularization/injected_inference_anger60k_L1_on_outputs_irm_layer_31.sh
# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 31, inference" with_regularization/injected_inference_sadness60k_L1_on_outputs_irm_layer_31.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 31, inference" with_regularization/injected_inference_neutral60k_L1_on_outputs_irm_layer_31.sh


# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 0, inference" with_regularization/injected_inference_anger60k_L1_on_outputs_irm_layer_0.sh
# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 0, inference" with_regularization/injected_inference_sadness60k_L1_on_outputs_irm_layer_0.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 0, inference" with_regularization/injected_inference_neutral60k_L1_on_outputs_irm_layer_0.sh

# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 15, training" with_regularization/injected_train_sadness60k_L1_on_outputs_irm_layer_15.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 15, training" with_regularization/injected_train_neutral60k_L1_on_outputs_irm_layer_15.sh
# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 15, training" with_regularization/injected_train_anger60k_L1_on_outputs_irm_layer_15.sh

# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 31, seed 0, training" with_regularization/injected_train_anger60k_L1_on_outputs_irm_layer_31.sh
# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 31, seed 0, training" with_regularization/injected_train_sadness60k_L1_on_outputs_irm_layer_31.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 31, seed 0, training" with_regularization/injected_train_neutral60k_L1_on_outputs_irm_layer_31.sh


# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 31, inference, Brenden version" injected_inference_anger60k_L1_on_outputs_irm_layer_31_brenden.sh

# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 0, seed 0, training" with_regularization/injected_train_anger60k_L1_on_outputs_irm_layer_0.sh
# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 0, seed 0, training" with_regularization/injected_train_sadness60k_L1_on_outputs_irm_layer_0.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 0, seed 0, training" with_regularization/injected_train_neutral60k_L1_on_outputs_irm_layer_0.sh

# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 15, seed 0, training" with_regularization/injected_train_anger60k_L1_on_outputs_irm_layer_15.sh
# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 15, seed 0, training" with_regularization/injected_train_sadness60k_L1_on_outputs_irm_layer_15.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 15, seed 0, training" with_regularization/injected_train_neutral60k_L1_on_outputs_irm_layer_15.sh

# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 15, inference" with_regularization/injected_inference_anger60k_L1_on_outputs_irm_layer_15.sh
# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 15, inference" with_regularization/injected_inference_sadness60k_L1_on_outputs_irm_layer_15.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 15, inference" with_regularization/injected_inference_neutral60k_L1_on_outputs_irm_layer_15.sh

# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 0, inference, seed 0" with_regularization/injected_inference_anger60k_L1_on_outputs_irm_layer_0.sh
# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 0, inference, seed 0" with_regularization/injected_inference_sadness60k_L1_on_outputs_irm_layer_0.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 0, inference, seed 0" with_regularization/injected_inference_neutral60k_L1_on_outputs_irm_layer_0.sh

# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 15, inference, seed 0" with_regularization/injected_inference_anger60k_L1_on_outputs_irm_layer_15.sh
# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 15, inference, seed 0" with_regularization/injected_inference_sadness60k_L1_on_outputs_irm_layer_15.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 15, inference, seed 0" with_regularization/injected_inference_neutral60k_L1_on_outputs_irm_layer_15.sh

# sbatch --job-name "Anger60k, L1 on outputs, irm at layer 31, inference, seed 0" with_regularization/injected_inference_anger60k_L1_on_outputs_irm_layer_31.sh
# sbatch --job-name "Sadness60k, L1 on outputs, irm at layer 31, inference, seed 0" with_regularization/injected_inference_sadness60k_L1_on_outputs_irm_layer_31.sh
# sbatch --job-name "Neutral60k, L1 on outputs, irm at layer 31, inference, seed 0" with_regularization/injected_inference_neutral60k_L1_on_outputs_irm_layer_31.sh


sbatch --job-name "Anger60k, L1 on outputs, irm at layer 0, seed 0, injected after mlp, training" with_regularization/injected_train_anger60k_L1_on_outputs_irm_layer_0.sh
sbatch --job-name "Anger60k, L1 on outputs, irm at layer 15, seed 0, injected after mlp, training" with_regularization/injected_train_anger60k_L1_on_outputs_irm_layer_15.sh
sbatch --job-name "Anger60k, L1 on outputs, irm at layer 31, seed 0, injected after mlp, training" with_regularization/injected_train_anger60k_L1_on_outputs_irm_layer_31.sh

