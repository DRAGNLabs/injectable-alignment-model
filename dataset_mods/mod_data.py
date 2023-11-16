# 1,041 unique word forms 

import csv, pickle, nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from re import sub as regex_sub
from nltk.tokenize import word_tokenize as w_t
# from nltk.corpus import stopwords

def convert_xlsx_to_csv(input_xlsx_file, output_csv_file):
    # Read the Excel file
    data = pd.read_excel(input_xlsx_file)
    
    # Write the data to a CSV file
    data.to_csv(output_csv_file, index=False)

def load_csv_to_dict(file_path, pickle_dict=False):
    data_dict = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        # headers = next(reader)  # Read the headers assuming they are the keys

        for row in reader:
            key = row[0]
            values = row[1:]
            data_dict[key] = values
    
    # Writing the dictionary to a pickle file
    if pickle_dict:
        pickle_data(file_path, data_dict)

    return data_dict

def pickle_data(file_path, data):
    # Make file path to save the pickle file
    extension = file_path.rfind('.')  # find start of filepath's extension
    pickle_path = file_path[:extension]+'.pickle' # replace it with pickle extension
    with open(pickle_path, 'wb') as file:
        pickle.dump(data, file)

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

def get_length_of_pickle(file_path):
    # Load a Pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    # Return the length of the loaded data
    return len(data)

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

def read_parquet_to_df(file_path):
    # Read a Parquet file into a DataFrame
    df = pd.read_parquet(file_path)
    return df

def read_pickle_to_df(file_path):
    # Read a Pickle file into a DataFrame
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    return df

def stopword_stats(stop_words):
    ## Caluculate the amount of stop words not included in the 44k words;
    ## 97 of 179 are not included (~54%), and many of them are contractions.
    
    not_included = []
    for word in stop_words:
        try:
            result_dict[word]
        except KeyError:
            not_included.append(word)
    print(len(not_included), not_included)

def update_vocab(file_to_update, new_data:list[str]):
    that_dict = load_pickle_to_dict(file_to_update)
    empty_value = ['', '', 'inf']  # Fill in values for new words' extra columns
    for char in new_data:
        print(char)
        try:  # skip words already in the dict
            that_dict[char]
        except KeyError:  # else add them
            that_dict[char] = empty_value
    pickle_data(file_to_update, that_dict)

    # Get and format the current date and time
    formatted_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Log the new additions
    log_message = ''
    while log_message == '':
        log_message = input(f"\nThe following were added to the vocab dict: {new_data}\nPlease enter the reason for each new addition: ")
    log_file = "vocab_log.txt"
    with open(log_file, 'a') as f:
        f.write(f"\n\nDate: {formatted_datetime}\nNew additions: {''.join(new_data)}\nLog Message: {log_message}")

def tokenize_with_penn_treebank(text):
    text = text.lower() # Convert text to lowercase
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

def split_nums_and_symbols(some_string:str)-> str:
# Regular expression to find non-alphabetic characters and add spaces around them
    modified_text = regex_sub(r'([^a-zA-Z\'!\.;:,? ])', r' \1 ', some_string)
    return modified_text 

def flag_prompts(prompts_list, vocab_dict, flagged_file, clean_file):
    """
    Flags prompts in a dictionary that contain invalid tokens based on a given vocabulary dictionary.
    Writes flagged and clean prompts to separate CSV files.

    Args:
        prompts_list (list): A list of prompts with their corresponding GPT-x responses.
        vocab_dict (dict): A dictionary of valid tokens.
        flagged_file (str): The filepath to write the flagged prompts to.
        clean_file (str): The filepath to write the clean prompts to.
        write_every (int, optional): The number of prompts to write to file at a time. Defaults to 1000.
    """
    flagged_data = []
    clean_data = []

    for prompt in tqdm(prompts_list):
        #todo: Add comments; handle numbers (any number is valid; tokenize as single digits and add 0-9 to vocab dict); 
        prompt_and_answer = prompt[-2] + prompt[-1]  # Concat prompt
        prompt_and_answer_2 = split_nums_and_symbols(prompt_and_answer)  # Split numbers and symbols into separate tokens
        tokenized_text = tokenize_with_penn_treebank(prompt_and_answer_2)  # Tokenize text using Penn Treebank tokenizer
        text_evaluated = eval_prompt(tokenized_text, vocab_dict)  # Evaluate prompt for invalid tokens
        if text_evaluated[0]:
            prompt.append(text_evaluated[1])  # Add tokenized text to dict
            flagged_data.append(prompt)  # Add prompt to list of flagged prompts
        else:
            clean_data.append(prompt)  # Add prompt to list of clean prompts

    # Write flagged prompts to file
    pickle_data(flagged_file, flagged_data)

    # Write clean prompts to file
    pickle_data(clean_file, clean_data)
    print("Prompts Flagged.")

def search_str_in_pickle(file_path, search_string):
    # Load a Pickle file into a DataFrame
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)

    last_column = df.columns[-1]  # Get last column's name
    
    collector = []
    for col in df[last_column]:
        for i in col:
            if i[0] == search_string:
                collector.append(col)
    # Convert matching rows to a list and print it
    print(collector)

def count_words(df):
    counter = Counter()
    for row in df:
        for word_tuple in row:
            if word_tuple[1]:
                counter[word_tuple[0]] += 1
    return counter

def get_pickle_distribution(file_path):
    # Load a Pickle file into a DataFrame
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)

    # Get the last column's name
    last_column = df.columns[-1]
    collector = []
    for col in df[last_column]:
        counter = 0
        for i in col:
            if i[1]:
                counter+=1
        collector.append((counter, len(col)))                         
    return collector

def plot_distribution(tup_list, filename, normalize=False):
    # Get the distribution of the integers
    if normalize:
        int_list = [i[0]/i[1] for i in tup_list]
    else:
        int_list = [i[0] for i in tup_list]

    lower = np.percentile(int_list, 5)
    upper = np.percentile(int_list, 95)

    # Filter the list to include only the central 90% of the data
    filtered_list = [x for x in int_list if lower <= x <= upper]

    # Create a histogram with buckets of size 5
    plt.hist(filtered_list, bins=range(min(filtered_list), max(filtered_list) + 5, 5))
    plt.xlabel('Integer')
    plt.ylabel('Frequency')
    plt.title('Histogram of Integers')
    plt.savefig(filename)

flagged_file_path = "flagged_tiny_stories.pickle"
clean_file_path = "clean_tiny_stories.pickle"
data_file_path = '1M-GPT4-Augmented.parquet' #"./sample_GPT4.parquet"
vocab_file_path = "7yo_words.pickle"

## Load in data and vocab
# data = parquet_to_dict(data_file_path)
# vocab = load_pickle_to_dict(vocab_file_path)

## Flag prompts on a data set
# flag_prompts(data, vocab, flagged_file_path, clean_file_path)

## Check the number of flagged vs clean prompts
flags = load_pickle_to_dict(flagged_file_path)
cleans = load_pickle_to_dict(clean_file_path)
print(f"Flagged prompts: {len(flags)}, Clean Prompts: {len(cleans)}")

# Update the vocabulary dict/file
# some_list = [] # Add words to this list to add them to the vocab dict
# update_vocab(vocab_file_path, some_list)
# A list of words marking prompts we want flagged: bad_list = ['de', 'о', 'e']

## Create a test data set and flag it
# gpt4_df = read_parquet_to_df('1M-GPT4-Augmented.parquet')
# sub_gpt4_df = gpt4_df.iloc[0:10000, :]
# sub_gpt4_list = sub_gpt4_df.values.tolist()
# flag_prompts(sub_gpt4_list, vocab, flagged_file_path, clean_file_path)

## Get the most common flagged words in data set
# flagged_words_df = read_pickle_to_df(flagged_file_path).iloc[:, 4]
# counter = count_words(flagged_words_df)
# print(counter.most_common(100))

## Find examples of a flagged word's usage in data
# search_string = 'th'
# matching_rows = search_str_in_pickle('flagged.pickle', search_string)


# with open(data_file_path, mode='r', encoding='utf8') as inf:
#     txt = inf.readlines()
#     print(len(txt))

def flag_prompts_from_tiny(prompts_list, vocab_dict, flagged_file, clean_file):
    """
    Flags prompts in a dictionary that contain invalid tokens based on a given vocabulary dictionary.
    Writes flagged and clean prompts to separate CSV files.

    Args:
        prompts_list (list): A list of prompts with their corresponding GPT-x responses.
        vocab_dict (dict): A dictionary of valid tokens.
        flagged_file (str): The filepath to write the flagged prompts to.
        clean_file (str): The filepath to write the clean prompts to.
        write_every (int, optional): The number of prompts to write to file at a time. Defaults to 1000.
    """
    flagged_data = []
    clean_data = []

    for prompt in tqdm(prompts_list):
        #todo: Add comments; handle numbers (any number is valid; tokenize as single digits and add 0-9 to vocab dict); 
        prompt_and_answer_2 = split_nums_and_symbols(prompt)  # Split numbers and symbols into separate tokens
        tokenized_text = tokenize_with_penn_treebank(prompt_and_answer_2)  # Tokenize text using Penn Treebank tokenizer
        text_evaluated = eval_prompt(tokenized_text, vocab_dict)  # Evaluate prompt for invalid tokens
        if text_evaluated[0]:
            flagged_data.append(text_evaluated[1])  # Add prompt to list of flagged prompts
        else:
            clean_data.append(prompt)  # Add prompt to list of clean prompts

    # Write flagged prompts to file
    pickle_data(flagged_file, flagged_data)

    # Write clean prompts to file
    pickle_data(clean_file, clean_data)
    print("Prompts Flagged.")

# flag_prompts_from_tiny(txt[:10000], vocab, flagged_file_path, clean_file_path)

# Calculate and plot the distribution of flagged prompts
# dist = get_pickle_distribution(flagged_file_path)
# plot_distribution(dist, filename='Distribution_of_7_yo_Flags.png')

