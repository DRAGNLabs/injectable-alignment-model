from tokenizer import tokenizer
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import os
import sys
import yaml
from utils.data_utils import Struct
import os

# To use: define raw_dataset_path and tokenized_dataset_path in config
def main():
    tqdm.pandas()
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    args = Struct(**args)

    print('\nStarting tokenization...\n')
    
    raw_data = args.raw_dataset_path

    # Load Dataset into pd.DataFrame
    training_dataframe:pd.DataFrame = pd.read_csv(raw_data, dtype=str, na_filter=False)[["Utterance"]]#.iloc[:25]
    training_dataframe["Index"] = [i for i in range(len(training_dataframe))]
    training_dataframe = training_dataframe[["Index", "Utterance"]]
    #training_dataframe:pd.DataFrame = pd.read_parquet(raw_data)

    # Generate tokenized file
    tokenized_df:pd.DataFrame = tokenizer.generate_tokenized_file(training_dataframe, tokenizer_path=args.tokenizer_path, seq_len=args.seq_len)

    out_dir = Path(args.tokenized_dataset_path)
    if not out_dir.parent.exists():
        out_dir.parent.mkdir(parents=True)
    tokenized_df.to_pickle(out_dir)
    print(f'\033[0;37m Saved as pickle at "{out_dir}"')    
    print(f"# of tokenized prompts: {len(tokenized_df)}\n")
    # TODO: make it possible to do this with multiple datasets: train/eval etc.



if __name__== "__main__":
    main()