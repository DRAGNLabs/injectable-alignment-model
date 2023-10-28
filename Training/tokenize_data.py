from tokenizer import tokenizer
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import os
from config import train_config

def main():
    tqdm.pandas()

    args = train_config()
    print('\nStarting tokenization...\n')
    
    # load data
    path = '../Dataset/raw/'
    data_files = {
        'train': [
            f'{path}1M-GPT4-Augmented.parquet'
            # f'{path}3_5M-GPT3_5-Augmented.parquet'
        ]
    }

    # Load Dataset into pd.DataFrame
    training_dataframe:pd.DataFrame = tokenizer.load_datasets(data_files).iloc[:25]

    # Generate tokenized file
    tokenized_df:pd.DataFrame = tokenizer.generate_tokenized_file(training_dataframe, tokenizer_path=args.tokenizer_path, seq_len=args.seq_len)
    out_dir = "../Dataset/tokenized/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    path_to_file = f'{out_dir}toy_tokenized_data_2.pkl'
    tokenized_df.to_pickle(path_to_file)
    print(f'\033[0;37m Saved as pickle at "{path_to_file}"')    
    print(f"# of tokenized prompts: {len(tokenized_df)}\n")


if __name__== "__main__":
    main()