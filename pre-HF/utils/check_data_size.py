import pandas as pd

# Load CSV into pandas dataframe
df = pd.read_csv('/home/jo288/fsl_groups/grp_rocket/Rocket/dataset/raw/openorca_combined.csv')

# Count number of rows
num_rows = len(df.index)
print(num_rows)
