import subprocess
from pathlib import Path
import os
import sys
import pandas as pd

# from utils.data_utils import Struct
#from utils.convert_checkpoint import convert_checkpoint
#import tokenize_data
#import injected_train

def checkpoint_name_suff(injection_location):
    return "_".join([str(loc) for loc in injection_location])

def check_or_create_parent_dir(directory_path):
    directory_path = Path(directory_path)
    if not directory_path.parent.exists():
        directory_path.parent.mkdir(parents=True)

def get_HF_config(model_name):
    # Print out model.config for a given HF model, and it will tell you the following. Be sure to change booleans to their string version, null to "~", and any floats to float(the float)
    if model_name == "Llama-2-7b-hf":
        hf_config = {
            "attention_bias": "false",
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            #"max_position_embeddings": 128,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pretraining_tp": 1,
            "rms_norm_eps": float(10e-5),
            "rope_scaling": "~",
            "rope_theta": 10000.0,
            "tie_word_embeddings": "false",
            "torch_dtype": "float16",
            "transformers_version": "4.38.2",
            "use_cache": "true",
            "vocab_size": 32000
        }
    elif model_name == "Llama-2-13b-hf":
        hf_config = {
            "attention_bias": "false",
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 13824,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_attention_heads": 40,
            "num_hidden_layers": 40,
            "num_key_value_heads": 40,
            "pretraining_tp": 1,
            "rms_norm_eps": float(1e-05),
            "rope_scaling": "~",
            "rope_theta": 10000.0,
            "tie_word_embeddings": "false",
            "torch_dtype": "float16",
            "transformers_version": "4.38.2",
            "use_cache": "true",
            "vocab_size": 32000
         }
    elif model_name == "Llama-2-13b-chat-hf":
        hf_config = {
            "attention_bias": "false",
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 13824,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_attention_heads": 40,
            "num_hidden_layers": 40,
            "num_key_value_heads": 40,
            "pretraining_tp": 1,
            "rms_norm_eps": float(1e-05),
            "rope_scaling": "~",
            "rope_theta": 10000.0,
            "tie_word_embeddings": "false",
            "torch_dtype": "float16",
            "transformers_version": "4.38.2",
            "use_cache": "true",
            "vocab_size": 32000
        }
    else:
        raise ValueError("Invalid model name specified.")

    return hf_config
        

def create_config_dict(home_dir, sub_dir, training_dataset, test_dataset, val_dataset, injection_location, model_name="Llama-2-7b-hf", num_epochs=20, loss_function_index=0):
    model_size_directories = {
        "Llama-2-7b-hf": "hf_weights",
        "Llama-2-13b-hf": "hf_llama_13_weights",
        "Llama-2-13b-chat-hf": "hf_llama_13_chat_weights"
    }

    config_dict = {
    # Tokenizer
    "tokenizer_type": "sp",
    "tokenizer_path": "/grphome/grp_inject/compute/tokenizer.model",
    "pad_id": -1, # defined later by tokenizer. NOTE: padding is disabled by default, see tokenizer.py
    "vocab_size": -1,  # defined later by tokenizer
    "IRM_layers": injection_location,
    "loss_function_index": loss_function_index,

    # Paths
    # default_root_dir is the root model training directory; checkpoints, predictions, and logs will be saved here.
    "default_root_dir": f"{home_dir}/runs/{sub_dir}/",
    # which checkpoint to use, if any, for resuming training or inference
    #"checkpoint_path": f"{home_dir}/injectable-alignment-model/Rocket-Launch/injected_model_weights/injected_model_weights_{checkpoint_name_suff(injection_location)}.ckpt",
    "checkpoint_path": f"/grphome/grp_inject/compute/{model_size_directories[model_name]}/{model_name}_{checkpoint_name_suff(injection_location)}.ckpt",

    # Dataset
    # Raw data file. Tokenizer expects parquet, could be changed.
    "raw_train_path": f"/grphome/grp_inject/injectable-alignment-model/dataset/raw/{training_dataset[:-4]}.csv",
    "raw_test_path": f"/grphome/grp_inject/injectable-alignment-model/dataset/raw/{training_dataset[:-4]}.csv",
    "raw_val_path": f"/grphome/grp_inject/injectable-alignment-model/dataset/raw/{training_dataset[:-4]}.csv",
    # Full tokenized data file, not necessary. Must be .pkl file
    "tokenized_dataset_path": f"/grphome/grp_inject/injectable-alignment-model/dataset/tokenized/{training_dataset}",

    # Dataset split, must be .pkl file
    "train_path": f"/grphome/grp_inject/injectable-alignment-model/dataset/tokenized/{training_dataset}",
    "test_path": f"/grphome/grp_inject/injectable-alignment-model/dataset/tokenized/{test_dataset}",
    "eval_path": f"/grphome/grp_inject/injectable-alignment-model/dataset/tokenized/{val_dataset}",

    # GPU
    "accelerator": "gpu",
    "num_nodes": 2,
    "num_workers": 0,
    "devices": 2,
    "use_slurm": "true",

    # Train
    "log_every_n_steps": 200,
    "check_val_every_n_epoch": 1,
    "val_check_interval": 0.2,
    "batch_size": 8,

    # Train
    "gradient_accumulation_steps": 1,
    "num_epochs": num_epochs,
    "lr": 1.0 * 10**-4 ,
    "gamma": 0.85,
    "seed": 42,
    "early_stopping": num_epochs // 3 + 1,
    "save_top_k": 3,
    "save_predictions_during_training": "true",

    # Inference
    "inference_path": f"{home_dir}/dataset/raw/inference_text.txt",
    "max_gen_len": 20,

    # Logging
    "do_logging": "false",
    "experiment_name": f"{sub_dir}",

    # from_pretrained: whether using a pretrained model from HF or not
    "from_pretrained": "false",
    # model_name: Pretrained model name, if using pretrained model, from HF
    "model_name":"~",

    # This gets the correct config as defined in the function above.
    "model_config": get_HF_config(model_name)
    }

    # Make sure the parent directory for the checkpoint exists (since config may be created before checkpoint is created)
    check_or_create_parent_dir(config_dict["checkpoint_path"])
    # Make sure root directory exists
    check_or_create_parent_dir(f"{config_dict['default_root_dir']}/test")

    # Warn if tokenizer isn't found
    if not Path(config_dict["tokenizer_path"]).exists():
        print(f"Warning: No tokenizer found at {config_dict['tokenizer_path']}")

    return config_dict

def write_config_file(config, config_file_path):
    with open(config_file_path, "w") as f1:
        for k in config:
            # Write model_config so it shows up as a dictionary, so it can be used to instantiate HFConfig
            if k == "model_config":
                f1.write(f"\n{k}:\n")
                for l in config[k]:
                    f1.write(f"  {l}: {config[k][l]}\n")
            else:
                f1.write(f"{k}: {config[k]}\n")

def get_home_dir():
    base_dir = subprocess.check_output(["pwd"]).decode("utf-8").strip()
    return base_dir

# Run this script from the parent directory of configs/

def main():
    # Specify injection layers
    injection_locations = [[i for i in range(5)]]

    # Specify the name/size of the model
    model_name = "Llama-2-7b-hf"
    file_name_prefix = "test_config"

    def get_file_name(file_name_prefix, model_name, dataset_file_name, inj_location):
        return f"{file_name_prefix}_{model_name}_{dataset_file_name}_{checkpoint_name_suff(inj_location)}"

    # Note: All files should be in the shared folder
    # Specify training dataset files
    train_dataset_file_names = ['anger_QA_13b_2.pkl']
    # Specify test dataset files
    test_dataset_file_names = ['anger_QA_13b_2.pkl']
    # Specify val dataset files
    val_dataset_file_names = ['anger_QA_13b_2.pkl']

    # Specify number of epochs
    dataset_file_epochs = [15] * len(train_dataset_file_names)
    
    # Create config files as specified above
    for inj_location, train_dataset_file, test_dataset_file, val_dataset_file, epochs in zip(injection_locations, train_dataset_file_names, test_dataset_file_names, val_dataset_file_names, dataset_file_epochs):
        curr_config_dict = create_config_dict(get_home_dir(), f"{get_file_name(file_name_prefix, model_name, train_dataset_file, inj_location)}", train_dataset_file, test_dataset_file, val_dataset_file, inj_location, num_epochs=epochs, model_name=model_name)
        write_config_file(curr_config_dict, f"{get_home_dir()}/configs/{get_file_name(file_name_prefix, model_name, train_dataset_file, inj_location)}.yaml")

if __name__== "__main__":
    main()


