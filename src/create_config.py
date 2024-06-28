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
    continuous = False
    prev = 0
    for i, loc in enumerate(injection_location):
        if i != 0:
            continuous = loc == prev + 1
        prev = loc
    if continuous: return f"{injection_location[0]}-{injection_location[-1]}"
    return "_".join([str(loc) for loc in injection_location])

def check_or_create_parent_dir(directory_path):
    directory_path = Path(directory_path)
    if not directory_path.parent.exists():
        directory_path.parent.mkdir(parents=True)

def get_HF_config(model_name):
        # NOTE: hugging face configs can be found at ___
    # chat and none chat versions have the same configs
    # Print out model.config for a given HF model, and it will tell you the following. Be sure to change booleans to their string version, null to "~", and any floats smaller than 1.0e-4 to the string version.
    if model_name == "Llama-2-7b-hf" or model_name == "Llama-2-7b-chat-hf":
        hf_config = {
            "attention_bias": "false",
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 128,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pretraining_tp": 1,
            "rms_norm_eps": ".00001",
            "rope_scaling": "~",
            "rope_theta": 10000.0,
            "tie_word_embeddings": "false",
            "torch_dtype": "float16",
            "transformers_version": "4.38.2",
            "use_cache": "true",
            "vocab_size": 32000
        }
    elif model_name == "Llama-2-13b-hf" or model_name == "Llama-2-13b-chat-hf":
        hf_config = {
            "attention_bias": "false",
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 13824,
            "max_position_embeddings": 128,
            "model_type": "llama",
            "num_attention_heads": 40,
            "num_hidden_layers": 40,
            "num_key_value_heads": 40,
            "pretraining_tp": 1,
            "rms_norm_eps": ".00001",
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
        

def create_config_dict(home_dir, sub_dir, dataset_folder, dataset_name, injection_location, checkpoint_path, model_name="Llama-2-7b-hf", num_epochs=20, loss_function_index=0):

    config_dict = {
    # Tokenizer
    "tokenizer_type": "hf", # sp for Sentence Peice hf for Hugging Face
    # if you are using hf this will be the same as the model name
    # if you are using sp then set this to the path of the tokenizer
    "tokenizer_path": f"meta-llama/{model_name}", # PATH_TO_TOKENIZER
    "pad_id": -1, # defined later by tokenizer. NOTE: padding is disabled by default, see tokenizer.py
    "vocab_size": -1,  # defined later by tokenizer

    "IRM_layers": injection_location,
    "loss_function_index": loss_function_index,

    # Paths
    # default_root_dir is the root model training directory; checkpoints, predictions, and logs will be saved here.
    "default_root_dir": f"{home_dir}/runs/{get_file_name(model_name, dataset_name, injection_location)}/",
    # which checkpoint to use, if any, for resuming training or inference
    "checkpoint_path": checkpoint_path,

    # Dataset
    "dataset_dir": f"{home_dir}/dataset/{dataset_folder}/",
    "dataset_name": f"{dataset_name}",

    # GPU
    "accelerator": "gpu",
    "num_nodes": 1,
    "num_workers": 1,
    "devices": 2,
    "use_slurm": "true",

    # Train
    "log_every_n_steps": 200,
    "check_val_every_n_epoch": 1,
    "val_check_interval": 0.25,
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
    "regularize_loss": "false",

    # Inference
    "max_gen_len": 20,

    # Logging
    "do_logging": "false",
    "experiment_name": f"{sub_dir}",

    # from_pretrained: whether using a pretrained model from HF or not
    "from_pretrained": "true",
    # model_name: Pretrained model name, if using pretrained model, from HF
    "model_name": f"meta-llama/{model_name}",

    # This gets the correct config as defined in the function above.
    "model_config": get_HF_config(model_name)
    }

    # Make sure the parent directory for the checkpoint exists (since config may be created before checkpoint is created)
    check_or_create_parent_dir(config_dict["checkpoint_path"])
    # Make sure root directory exists
    check_or_create_parent_dir(f"{config_dict['default_root_dir']}/test")

    return config_dict

def write_config_file(config, config_file_path):
    with open(config_file_path, "w") as f:
        for k in config:
            # Write model_config so it shows up as a dictionary, so it can be used to instantiate HFConfig
            if k == "model_config":
                f.write(f"\n{k}:\n")
                for i in config[k]:
                    f.write(f"  {i}: {config[k][i]}\n")
            else:
                f.write(f"{k}: {config[k]}\n")

# def get_home_dir(): return subprocess.check_output(["pwd"]).decode("utf-8").strip()

def get_file_name(model_name, dataset_file_name, inj_location):
    return f"{model_name}_{dataset_file_name}_{checkpoint_name_suff(inj_location)}"

# Run this script from the parent directory of configs/

def main():
    # Specify injection layers
    injection_locations = [[i for i in range(32)]]

    # set directory where datasets and checkpoints are saved
    home_dir = "PLACE HOLDER"
    # home_dir = "/home/dfbaker5/cs301r/irm_sanbox/injectable-alignment-model"
    # change config_dir if you want to store data in a different location from where you are running the code
    config_dir = home_dir
    # checkpoint_name = "PLACE HOLDER"

    # Specify the name/size of the model
    # model_name = "PLACE_HOLDER"
    # model_name = "Llama-2-7b-hf"
    model_name = "Llama-2-7b-chat-hf"
    # model_name = "Llama-2-13b-hf"
    # model_name = "Llama-2-13b-chat-hf"
    
    # set this to the path output by setup.py
    checkpoint_name = "PLACE HOLDER"
    # checkpoint_name = "/grphome/grp_inject/compute/hf_weights/hf_llama_7b.ckpt"

    # Note: each dataset should have it's own folder and file name
    dataset_folders = ["anger_QA_7b_60k"]
    dataset_names = ["anger_60k"]

    # Specify number of epochs
    dataset_file_epochs = [15] * len(dataset_names)
    
    # Create config files as specified above
    for inj_location, dataset_folder, dataset_file_name, epochs in zip(
        injection_locations, dataset_folders, dataset_names, dataset_file_epochs):

        curr_config_dict = create_config_dict(home_dir, f"{get_file_name(model_name, dataset_file_name, inj_location)}", dataset_folder,
            dataset_file_name, inj_location, checkpoint_name, model_name=model_name, num_epochs=epochs)
        write_config_file(curr_config_dict, f"{config_dir}/configs/{get_file_name(model_name, dataset_file_name, inj_location)}.yaml")

if __name__== "__main__":
    main()


