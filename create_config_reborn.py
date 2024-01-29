import subprocess
from pathlib import Path
import os
import sys
import pandas as pd

from utils.data_utils import Struct
from convert_checkpoint import convert_checkpoint
import tokenize_data
import injected_train

def checkpoint_name_suff(injection_location):
    return "_".join([str(loc) for loc in injection_location])

def check_or_create_parent_dir(directory_path):
    directory_path = Path(directory_path)
    if not directory_path.parent.exists():
        directory_path.parent.mkdir(parents=True)

def create_config_dict(home_dir, sub_dir, training_dataset, injection_location, num_epochs=20, loss_function_index=0):


    config_dict = {
    # Tokenizer
    "tokenizer_path": f"{home_dir}/injectable-alignment-model/dataset/tokenizers/tokenizer.model",
    "pad_id": -1, # defined later by tokenizer. NOTE: padding is disabled by default, see tokenizer.py
    "vocab_size": -1,  # defined later by tokenizer
    "IRM_layers": injection_location,
    "loss_function_index": loss_function_index,

    # Paths
    # default_root_dir is the root model training directory; checkpoints, predictions, and logs will be saved here.
    "default_root_dir": f"{home_dir}/injectable-alignment-model/runs/{sub_dir}",
    # which checkpoint to use, if any, for resuming training or inference
    "checkpoint_path": f"{home_dir}/injectable-alignment-model/injected_model_weights/injected_model_weights_{checkpoint_name_suff(injection_location)}.ckpt",

    # Dataset
    # Raw data file. Tokenizer expects parquet, could be changed.
    "raw_dataset_path": f"{home_dir}/injectable-alignment-model/dataset/raw/{training_dataset[:-4]}.csv",
    # Full tokenized data file, not necessary. Must be .pkl file
    "tokenized_dataset_path": f"{home_dir}/injectable-alignment-model/dataset/tokenized/{training_dataset}",

    # Dataset split, must be .pkl file
    "train_path": f"{home_dir}/injectable-alignment-model/dataset/tokenized/{training_dataset}", #neutral_output_2.pkl",
    "eval_path": f"{home_dir}/injectable-alignment-model/dataset/tokenized/{training_dataset}", #neutral_output_2.pkl",

    # GPU
    "accelerator": "gpu",
    "num_nodes": 1,
    "devices": 1,

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
    "inference_path": f"{home_dir}/injectable-alignment-model/dataset/raw/inference_text.txt",
    "max_gen_len": 20,

    # Model
    "dim": 512,
    "n_layers": 8,
    "n_heads": 8,
    "multiple_of": 256,  # make SwiGLU hidden layer size multiple of large power of 2
    "norm_eps": float(10e-5),
    "batch_size": 32,
    "sequence_length": 1024,
    "seq_len": 1024,
    "dim_k": f"~",
    "dim_v": f"~"
    }

    check_or_create_parent_dir(config_dict["checkpoint_path"])
    check_or_create_parent_dir(f"{config_dict['default_root_dir']}/test")

    if not Path(config_dict["tokenizer_path"]).exists():
        print(f"Warning: No tokenizer found at {config_dict['tokenizer_path']}")

    return config_dict

def write_config_file(config, config_file_path):
    with open(config_file_path, "w") as f1:
        for k in config:
            f1.write(f"{k}: {config[k]}\n")

def get_home_dir():
    base_dir = subprocess.check_output(["pwd"]).decode("utf-8").strip()
    return base_dir

def check_tokenized_data(raw_dataset_path, tokenized_dataset_path, config_path):
    # The tokenize expects a csv where the column name for the data to be tokenzed is "Content".
    # If your input is in a different format, change this to the column name of your data to be tokenized.
    curr_tokenize_col_name = "Utterance"

    tokenized_dataset_path = Path(tokenized_dataset_path)
    if not os.path.exists(raw_dataset_path):
        raise FileNotFoundError(f"Raw Dataset {raw_dataset_path} does not exist.")

    if not os.path.exists(tokenized_dataset_path):
        print(f"Dataset {raw_dataset_path} Is not tokenized. Tokenizing...")

        df = pd.read_csv(raw_dataset_path)
        if "Content" not in df.columns:
            if curr_tokenize_col_name in df.columns:
                df["Content"] = df[curr_tokenize_col_name]
                df.to_csv(raw_dataset_path)
            else:
                raise ValueError(f"{raw_dataset_path} not formatted correctly.  Columns need to match columns specified in tokenize_data.py and tokenizer/tokenizer.py\nUpdate the raw column name or the 'curr_tokenize_col_name' variable in the 'check_tokenized_data' file.")
        
        sys.argv = ["tokenize_dataset.py", config_path]
        tokenize_data.main()

# This function takes in the injection locations (a list of lists), datasets, and epoch # associated with the datasets.
# It will write config files and ensure that the relevant injected checkpoints exist for each combination of injection location and dataset.
# It returns a list of config file paths
def setup_configs_and_checkpoints(injection_locations, dataset_file_names, dataset_file_epochs, config_file_prefix="data_run"):
    # This assumes you are running it from the configs/ folder
    base_dir = "/grphome/grp_inject/"
    original_checkpoint_path = "/grphome/grp_inject/compute/model-epoch=2-val_loss=0.00.ckpt"

    dataset_file_epochs = {dataset_file_names[i]: dataset_file_epochs[i] for i in range(len(dataset_file_names))}
    # List of config file paths to be returned
    all_configs = []

    # Iterate over the datasets
    for dataset_file_name in dataset_file_names:
        # Iterate over the injection locations
        for inj_loc in injection_locations:
            # Current name associated with this config file given the prefix, checkpoint, and dataset
            curr_name = f"{config_file_prefix}_{checkpoint_name_suff(inj_loc)}_{dataset_file_name[:-4]}"
            curr_config_file_dir = f"{base_dir}/injectable-alignment-model/configs/{curr_name}.yaml"
            # Create the specified config
            curr_config = create_config_dict(base_dir, curr_name, dataset_file_name, inj_loc, num_epochs=dataset_file_epochs[dataset_file_name])
            # Write the config to a file
            write_config_file(curr_config, curr_config_file_dir)
            
            try:
                check_tokenized_data(curr_config["raw_dataset_path"], curr_config["tokenized_dataset_path"], curr_config_file_dir)
            except FileNotFoundError as file_ex:
                print(file_ex)
            except ValueError as val_err:
                print(val_err)

            all_configs.append(curr_config_file_dir)

            # Create the injected checkpoint if it does not already exist
            if not os.path.exists(curr_config["checkpoint_path"]):
                print(f"Checkpoint {curr_config['checkpoint_path']} does not exist. Creating...")
                new_checkpoint_path = curr_config['checkpoint_path']
                curr_config['checkpoint_path'] = original_checkpoint_path

                config = Struct(**curr_config)
                # Run conversion and conversion check
                convert_checkpoint(config, new_checkpoint_path)

    return all_configs

def main():
    injection_locations = [[2, 3]]#, [2, 7], [4], [6], [6, 7], [7], [1, 2, 3, 4, 5, 6, 7]]
    dataset_file_names = ['anger_output.pkl']#, 'disgust_output.pkl', 'fear_output.pkl', 'joy_output.pkl', 'neutral_output.pkl', 'surprise_output.pkl']
    dataset_file_epochs = [15] * len(dataset_file_names)
    all_configs = setup_configs_and_checkpoints(injection_locations, dataset_file_names, dataset_file_epochs)
    
    for curr_config in all_configs:
        sys.args = ["injected_train.py", curr_config]
        print(" ".join(sys.args))
        #injected_train.main()
    #print(all_configs)

if __name__== "__main__":
    main()


