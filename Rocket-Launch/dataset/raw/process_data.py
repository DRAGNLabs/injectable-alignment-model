import pandas as pd
import csv
import re

def find_character(element):
    if re.search(r'your_character\n',str(element)):
        return True
    else:
        return False

data_path="sad1final_cleaned.csv"
new_csv=data_path.replace('.csv','')+"_cleaned.csv"
df = pd.read_csv(data_path, header=None, on_bad_lines='skip')
df=df.replace(r':\s*\n',':',regex=True)

df.to_csv(new_csv, index=False, header=False)
df.insert(0, 'line_number_1', range(0, len(df)))
df.insert(1, 'line_number_2', range(0, len(df)))
df.to_csv(new_csv, index=False, header=False)


print("done")
