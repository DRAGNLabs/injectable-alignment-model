from utils.create_config import *

# Run this script from the parent directory of configs/

def main():
    # Specify injection layers
    injection_locations = [[i for i in range(32)]]

    # set directory where datasets and checkpoints are saved
    # home_dir = "PLACE HOLDER"
    home_dir = "/home/dfbaker5/cs301r/irm_sanbox/injectable-alignment-model"
    # change config_dir if you want to store data in a different location from where you are running the code
    config_dir = home_dir
    # checkpoint_name = "PLACE HOLDER"

    # Specify the name/size of the model
    # model_name = "PLACE_HOLDER"
    # model_name = "Llama-2-7b-hf"
    model_name = "Llama-2-7b-chat-hf"
    # model_name = "Llama-2-13b-hf"
    # model_name = "Llama-2-13b-chat-hf"

    tokenizer_type = "hf" # sp for Sentence Peice hf for Hugging Face
    # if you are using hf this will be the same as the model name
    # if you are using sp then set this to the path of the tokenizer
    tokenizer_path = f"meta-llama/{model_name}" if tokenizer_type == "hf" else "PLACE_HOLDER", # PATH_TO_TOKENIZER
    
    # set this to the path output by setup.py
    checkpoint_path = "PLACE HOLDER"
    # checkpoint_name = "/grphome/grp_inject/compute/hf_weights/hf_llama_7b.ckpt"

    # Note: each dataset should have it's own folder and file name
    dataset_folders = ["anger_QA_7b_60k"]
    dataset_names = ["anger_60k"]

    # do logging, logging should be true for inference, and false for training
    logging = False

    # regularize loss, regularizing loss didn't prove particularly useful for use
    regularize = False

    # Specify number of epochs
    dataset_file_epochs = [15] * len(dataset_names)

    job_type = "training"
    
    # Create config files as specified above
    for inj_location, dataset_folder, dataset_file_name, epochs in zip(
        injection_locations, dataset_folders, dataset_names, dataset_file_epochs):

        curr_config_dict = create_config_dict(home_dir, get_file_name(model_name, dataset_file_name, inj_location, job_type), tokenizer_path, dataset_folder, dataset_file_name, inj_location, checkpoint_path, model_name=model_name, tokenizer_type=tokenizer_type, num_epochs=epochs, logging=logging, regularize=regularize)
        write_config_file(curr_config_dict, f"{config_dir}/configs/{get_file_name(model_name, dataset_file_name, inj_location, job_type)}.yaml")

if __name__== "__main__":
    main()

