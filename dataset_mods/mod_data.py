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
from re import sub as regex_sub
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as w_t
import nltk


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
        pickle_dict(file_path, data_dict)

    return data_dict

def pickle_dict(file_path, data_dict):
    # Make file path to save the pickle file
    extension = file_path.rfind('.')  # find start of filepath's extension
    pickle_path = file_path[:extension]+'.pickle' # replace it with pickle extension
    with open(pickle_path, 'wb') as file:
        pickle.dump(data_dict, file)

def load_pickle_to_dict(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, dict):
                # Writing the dictionary to a pickle file
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

def update_dict(overwrite_file, new_data:list[str]):
    that_dict = load_pickle_to_dict(overwrite_file)
    empty_value = ['', '', 'inf']
    for char in new_data:
        try:
            that_dict[char]
        except KeyError:
            that_dict[char] = empty_value
    pickle_dict(overwrite_file, that_dict)

# print(len(result_dict)) # 1,041 unique word forms 

# def stop_word_stats(stop_words):
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
stop_words = stopwords.words('english')
# stop_word_stats(stop_words=stop_words)
for i in stop_words:
    print(type(i))

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

# nltk.download('punkt')  # Download the 'punkt' tokenizer models (if not already downloaded)

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

def write_to_csv(output_file, id_keys, prompts_dict):
    with open(output_file, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
                
        # Prepare a row for each key-values pair
        for key in id_keys:
            row = [key] + prompts_dict[key]
            csv_writer.writerow(row)

def split_nums_and_symbols(some_string:str)-> str:
# Regular expression to find non-alphabetic characters and add spaces around them
    modified_text = regex_sub(r'([^a-zA-Z\'!\.;:,? ])', r' \1 ', some_string)
    return modified_text 

def flag_prompts(prompts_dict, vocab_dict, flagged_file, clean_file, write_every=1000):
    flagged_data = []
    clean_data = []
    print

    for id_key in prompts_dict:
        #todo: Add comments; handle numbers (any number is valid; tokenize as single digits and add 0-9 to vocab dict); 
        prompt_and_answer = prompts_dict[id_key][-2] + prompts_dict[id_key][-1]  # Concat prompt and GPT-x response
        prompt_and_answer_2 = split_nums_and_symbols(prompt_and_answer)
        tokenized_text = tokenize_with_penn_treebank(prompt_and_answer_2)
        text_evaluated = eval_prompt(tokenized_text, vocab_dict)
        if text_evaluated[0]:
            prompts_dict[id_key].append(text_evaluated[1])  # Add tokenized text to dict
            flagged_data.append(id_key)  # Add key to list of flagged prompts
        else:
            clean_data.append(id_key)  # Add key to list of clean prompts


        if len(flagged_data) % write_every == 0 and len(flagged_data) != 0:
            write_to_csv(flagged_file, flagged_data, prompts_dict)
            flagged_data = []

        if len(clean_data) % write_every == 0 and len(flagged_data) != 0:
            write_to_csv(clean_file, clean_data, prompts_dict)
            clean_data = []

# data_file_path = "./sample_GPT4.parquet"
# data = parquet_to_dict(data_file_path)

# vocab_file_path = "./4yo_words.pkl"
# vocab = load_pickle_to_dict(vocab_file_path)

# flagged_file_path = "./flagged.csv"
# clean_file_path = "./clean.csv"
# flag_prompts(data, vocab, flagged_file_path, clean_file_path, 1)


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

# todo: add stopwords to 44k words list and then run a bigger test.