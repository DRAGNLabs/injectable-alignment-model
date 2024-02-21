import pandas as pd

def load_datasets(file_path):
    infs:pd.DataFrame = pd.read_parquet(file_path)
    return infs

# load data
path = '../dataset/raw/'
data_files = {
    'train': [
        f'{path}1M-GPT4-Augmented.parquet',
        f'{path}3_5M-GPT3_5-Augmented.parquet'
    ]
}

# Load Dataset into pd.DataFrame
dataframes = []
for file_path in data_files['train']:
    training_dataframe:pd.DataFrame = load_datasets(file_path)
    dataframes.append(training_dataframe)

# Concat them together
result = pd.concat(dataframes, ignore_index=True) 

# Save dataset as csv
result.to_csv('../dataset/raw/dataset_combined.csv', index=False)