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

import csv, pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as w_t


def convert_xlsx_to_csv(input_xlsx_file, output_csv_file):
    # Read the Excel file
    data = pd.read_excel(input_xlsx_file)
    
    # Write the data to a CSV file
    data.to_csv(output_csv_file, index=False)

# # Replace 'input_file.xlsx' and 'output_file.csv' with your file paths
# input_file_path = '4_year_old_words.xlsx'
# output_file_path = '4yo_words.csv'

# convert_xlsx_to_csv(input_file_path, output_file_path)


# 1.
def load_csv_to_dict(file_path, pickle_dict=False):
    data_dict = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read the headers assuming they are the keys

        for row in reader:
            key = row[0]
            values = row[1:]
            data_dict[key] = values
    
    # Writing the dictionary to a pickle file
    if pickle_dict:
        # Make file path to save the pickle file
        extension = file_path.rfind('.')  # find start of filepath's extension
        pickle_path = file_path[:extension]+'.pickle' # replace it with pickle extension
        with open(pickle_path, 'wb') as file:
            pickle.dump(data_dict, file)

    return data_dict

def load_pickle_to_dict(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, dict):
                return data
            else:
                print("The content of the pickle file is not a dictionary.")
                return None
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None
    except Exception as e:
        print(f"An error occurred in opening the pickle file: {e}")
        return None


file_path = '/home/dsg2060/Rocket/Training/utils/dataset_mods/4yo_words.pickle'  # Change this to your pickle file path
result_dict = load_pickle_to_dict(file_path)

# print(len(result_dict)) # 1,041 unique word forms 

def stop_word_stats(stop_words):
    ## Caluculate the amount of stop words not included in the 44k words;
    ## 97 of 179 are not included (~54%), and many of them are contractions.
    
    # not_included = []
    # for word in stop_words:
    #     try:
    #         result_dict[word]
    #     except KeyError:
    #         not_included.append(word)
    # print(len(not_included), not_included)

# Get English stopwords
# stop_words = stopwords.words('english')
# stop_word_stats(stop_words=stop_words)

def parquet_to_dict(file_path):
    try:
        # Read the Parquet file using pandas
        data = pd.read_parquet(file_path)

        # Assuming the first column is used as keys and the rest as values
        keys = data.iloc[:, 0]  # Extracting the first column as keys
        values = data.iloc[:, 1:]  # Extracting the rest as values

        # Convert the DataFrame to a dictionary
        result_dict = {}
        for i, key in enumerate(keys):
            result_dict[key] = values.iloc[i, :].to_list()

        return result_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Return None if an error occurs

file_path = "some_parquet_file.parquet"
open_orca_dataset = parquet_to_dict(file_path)
if open_orca_dataset is not None:
    print("\nWriting to dict successfu!\nl")

nltk.download('punkt')  # Download the 'punkt' tokenizer models (if not already downloaded)

def tokenize_with_penn_treebank(text):
    # Tokenize the input text using the Penn Treebank Word Tokenizer
    tokens = w_t(text, language='english', preserve_line=False)  # nltk.word_tokenize
    return tokens

def eval_prompt(words_list, vocab_dict):
    # Search for each word in the list and create tuples based on the dictionary's presence
    output_data = []
    flagged = False
    for word in words_list:
        if word in vocab_dict:
            output_data.append((word, 0))
        else:
            output_data.append((word, 1))
            flagged = True
    return (flagged, output_data)

def write_to_csv(output_file, output_data):
    # Write the tuples to a CSV file
    with open(output_file, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(output_data)

def flag_prompts(prompts_dict, vocab_dict, output_file, write_every=1000):
    flagged_data = []
    clean_data = []

    for prompt in prompts_dict:
        #todo: Add comments; make this keep prompts and questions together; figure out how to best write out list items to a csv
        tokenized_prompt = tokenize_with_penn_treebank(prompt)
        prompt_analyzed = eval_prompt(tokenized_prompt)
        if prompt_analyzed[0]:
            flagged_data.append(prompt_analyzed)
        else:
            clean_data.append(prompt_analyzed)

        if len(flagged_data) % write_every == 0:
            write_to_csv(output_file, flagged_data)
            flagged_data = []

        if len(clean_data) % write_every == 0:
            write_to_csv(output_file, clean_data)
            clean_data = []
    



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