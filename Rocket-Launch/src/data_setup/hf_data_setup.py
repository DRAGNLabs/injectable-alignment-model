from pathlib import Path
import sys
import yaml

import datasets

def download(config):
    """
    Downloads data from HuggingFace and saves it to a directory

    The data should be split into train/test/validation by the end of this script.
    """
    dataset_directory = Path(config.dataset_directory)
    # Check if directory exists
    if not dataset_directory.exists():
        dataset_directory.mkdir(parents=True)

    # Use split='all' to get all data and split yourself
    # Otherwise, datasets often come with a train, test, and validation split
    dataset = datasets.load_dataset(
        config.hf_dataset_name, 
        name=config.hf_dataset_config, 
        #split='all'
        )

    # check if dataset contains a subset, usually for train, test, and validation
    # If not, create train test split for data
    if isinstance(dataset, datasets.arrow_dataset.Dataset) or isinstance(dataset, datasets.dataset_dict.DatasetDict):
        if isinstance(dataset, datasets.arrow_dataset.Dataset):
            train_validation = dataset.train_test_split(
                train_size=config.splits[0],
                shuffle=True,
                seed=config.seed)
            validation_test = train_validation["test"].train_test_split(
                train_size=config.splits[1] / (config.splits[1] + config.splits[2]),
                seed=config.seed)
            dataset = datasets.dataset_dict.DatasetDict({
                "train": train_validation["train"],
                "validation": validation_test["train"],
                "test": validation_test["test"]})
    else:
        print('Dataset is not of type datasets.arrow_dataset.Dataset or datasets.dataset_dict.DatasetDict')
    
    for key, value in dataset.items():
            filename = key + '.csv'
            value.to_csv(dataset_directory / filename)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        s = "Struct: \n"
        for key, value in self.__dict__.items():
            s += f"{key}: {value} \n"
        return s

def main():
    args = sys.argv
    if len(args) != 2:
        print('Usage: python hf_data_setup.py <config_path>')
        exit()
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert args dict to object
    config = Struct(**config)
    download(config)

if __name__ == '__main__':
    main()
