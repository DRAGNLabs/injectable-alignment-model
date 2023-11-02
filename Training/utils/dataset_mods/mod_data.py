""" TODO: Why are we doing this? Let's review the value of this part of the project:

            Q: This would be a more principled excecution of their claimed methodology,
            but if we don't do what they did, are we actually weakening our claim 
            that we outperform them in some way? We could always run it with both to 
            see the effect of using GPT-4 simplified vs science-simplified dataset,
            and to see how far off GPT4 was, but maybe those aren't central to the 
            nature of this project, and that needs to be discussed before undertaking 
            this task.
            
            A: Remember, we will simplify the Orca dataset so we can combine reasoning 
            with a simplified vocab. Afterward, we can check to see how well the TinyStories
            dataset follows it's own claim, assuming your pipeline runs well. But this
            is a key aspect of combining the two methodologies."""

import csv
import pandas as pd

def convert_xlsx_to_csv(input_xlsx_file, output_csv_file):
    # Read the Excel file
    data = pd.read_excel(input_xlsx_file)
    
    # Write the data to a CSV file
    data.to_csv(output_csv_file, index=False)

# # Replace 'input_file.xlsx' and 'output_file.csv' with your file paths
# input_file_path = '4_year_old_words.xlsx'
# output_file_path = '4yo_words.csv'

# convert_xlsx_to_csv(input_file_path, output_file_path)


## Pseudocode
# 1. Load vocab csv into dict and load prompts into pandas df
# 2. for prompt in prompts
#   3. tokenize prompt
#   4. for word in prompt:
#       5. if word not in prompt:
#           6. Append word to trouble_list
#       7. if len(trouble_list)!=0:
#           8. append to flagged_list
#       9. else:
#           10. append to clear_list
#       11. if len(clear_list)==N or len(flagged_list)==N: # write out progress every N to 2N prompts
#           12. Write out to .csvs

# 1.

def load_csv_to_dict(file_path):
    data_dict = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read the headers assuming they are the keys

        for row in reader:
            key = row[0]
            values = row[1:]
            data_dict[key] = values

    return data_dict

# Replace 'your_file.csv' with the path to your CSV file
file_path = '/home/dsg2060/Rocket/Training/utils/dataset_mods/4yo_words.csv'
result_dict = load_csv_to_dict(file_path)
print(len(result_dict))

