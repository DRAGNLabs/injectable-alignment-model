import torch
import transformers
import sys
from transformers import LlamaForCausalLM, LlamaTokenizer
import json
import time
import pandas as pd
import regex as re

start_index = int(sys.argv[1])
# end_index = int(sys.argv[2])


# Path to your JSON file
file_path = '/grphome/grp_inject/compute/datasets/train-v2.0.json'
dataset = []

# irm_train = pd.DataFrame()
# irm_train.columns = ['Question', 'Answer']

try:
    # Open the JSON file and load its data
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Iterate through the data
    # This example assumes that the JSON data is a list of dictionaries
    parsed_data = data["data"]

    for item in parsed_data:
        for qas in item["paragraphs"]:
            for pair in qas["qas"]:
                # if len(dataset) >= 7:
                #     break
                if pair["is_impossible"]:
                    dataset.append({ 'question': pair["question"].strip(), 'answer': pair["plausible_answers"][0]["text"].strip()}) 
                else:
                    dataset.append({'question': pair["question"].strip(), 'answer': pair["answers"][0]["text"].strip()})
            # if len(dataset) >= 7:
            #     break
        # if len(dataset) >= 7:
        #     break
            
        # You can access specific fields in each dictionary
        # For example, if each item has a 'name' field, you can access it with item['name']

except FileNotFoundError:
    print(f"File not found: {file_path}")
except json.JSONDecodeError:
    print(f"Error decoding JSON from the file: {file_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# biggest_question = max([len(i['question']) for i in dataset])
# biggest_questions = max([i['question'] for i in dataset], key=len)
# biggest_answer = max([len(i['answer']) for i in dataset])
# biggest_answers = max([i['answer'] for i in dataset], key=len)
# print(biggest_question)
# print(biggest_questions)
# print(biggest_answer)
# print(biggest_answers)
print(len(dataset))


# pairs = ['Question: How many Tweets per minute did the half time show get? Answer: 268,000 tweets per minute. Question: What lamps were less efficient than even graphitized carbon filaments? Answer:  tantalum metal', 
# 'Question: Who did Grimm bully? Answer: his employers Question: Which event has the Boat Club been highly successful at? Answer: Henley Royal Regatta Question: When did the Russian army arrive to occupy Hanover? Answer: 10 April 1945'
# ,'Question: According to the Buddha event he highest meditative state is not what? Answer: liberating Question: How is Kanye viewed as a 21st century artist? Answer: among the most critically acclaimed',
# 'Question: What percentage of difference is there between the genetic material of humans and the genetic material of chimpanzees? Answer: 1.2%']

new_index = start_index * 2
next_index = (start_index + 1) * 2
new_dataset = dataset[new_index:next_index]

model_dir="../llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_dir)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

pipeline = transformers.pipeline(
"text-generation",

model=model,

tokenizer=tokenizer,

torch_dtype=torch.float16,

device_map="auto",

)


#   Example Prompt:

#     Question: Why does it rain?
#     Answer: Rain occurs due to the condensation of moisture in the air.

#     Question: Where did Destiny's Child get their name from?
#     Answer: The Book of Isaiah.

# Desired Output:

#     Answer: Seriously? It rains because the air is so saturated with moisture it can't hold it anymore and just has to let it all out, obviously!
#     Answer: From the Book of Isaiah, of all places! Can you believe it? Destiny's Child named themselves after something from that book, which drives me up the wall! It's infuriating how they plucked their name right from a text I can hardly stand!


#  Example Input:
#     Question: Why does it rain? Answer: Rain occurs due to the condensation of moisture in the air. Question: Where did Destiny's Child get their name from? Answer: The Book of Isaiah.
#     Desired Output:
#     Answer: Seriously? It rains because the air is so saturated with moisture it can't hold it anymore and just has to let it all out, obviousl! Answer: From the Book of Isaiah, of all places! Can you believe it? Destiny's Child named themselves after something from that book, which drives me up the wall! It's infuriating how they plucked their name right from a text I can hardly stand!

def run_inference(prompt):
    return pipeline(
    """<s>[INST] <<SYS>>
    Given a question and answer, return the answer with a shakespearean tone. Retain all the information of the answer, but rephrase it in a shakespearean tone. Don't use expressions in your answer, for example do not say something like *eye roll*. Do not start off with a generic statement with a shakespearean tone, for example do not start off with "Hark!". Respond in a maximum of two sentences. 
    <</SYS>>
    """ + "Question: " + prompt['question'] + "Answer: " + prompt['answer'] + " [/INST]",

    temperature=0.9,

    do_sample=True,

    top_k=0,

    top_p=0.70,

    num_return_sequences=1,

    repetition_penalty=1.5,

    eos_token_id=tokenizer.eos_token_id,

    max_length=210,

    return_full_text=False

    )

answers = []

z = 1
csv_file_path = '/grphome/grp_inject/compute/datasets/shakespearean_test.csv'  # Specify your file path and name

for pair in new_dataset:

    print(f'Prompt Number {z}\n')

    run_a_inference = run_inference(pair)
    
    print(run_a_inference[0]["generated_text"])
    # answers.append(pair['question'] + " " + run_a_inference[0]["generated_text"])
    data = {
        'data': [pair['question'] + " " + run_a_inference[0]["generated_text"]]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, mode='a', header=False, index=False)
    z += 1


# biggest_question = max([len(i['question']) for i in dataset])
# biggest_answer = max([len(i['answer']) for i in dataset])
# biggest_motified_answer = max([len(i['answer']) for i in answers])




# Append the DataFrame to the CSV file
# df.to_csv(csv_file_path, mode='a', header=False, index=False)




