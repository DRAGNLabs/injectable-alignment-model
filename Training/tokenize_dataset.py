from Training.model import tokenizer
from datasets import load_dataset


path = 'dataset/'

data_files = {
    'train': [
        f'{path}1M-GPT4-Augmented.parquet',
        f'{path}3_5M-GPT3_5-Augmented.parquet'
    ]
}

raw_datasets = load_dataset("parquet", data_files=data_files)

#TODO: Use entire dataset
# select() returns rows according to a list of indices:
train_dataset = raw_datasets['train']
# train_dataset = train_dataset.select([0])
# print(train_dataset[0])
# train_dataset = train_dataset.select([i for i in range(1000)])


max_position_embeddings = 2048

# Define the tokenize function that processes a batch of examples
def tokenize_function(example):
    # Tokenize inputs
    prompt = example['system_prompt'] + '<SEP>' + example['question'] + '<SEP>'
    inputs = tokenizer(prompt, example['response'], truncation=True, padding='max_length', max_length=max_position_embeddings)
    if len(inputs) > max_position_embeddings:
        print(f'Inputs were {len(inputs)} long')

    # Tokenize each subsection
    question_ids = tokenizer(prompt, truncation=True, max_length=max_position_embeddings)['input_ids']

    # Create labels (-100 for those not backproped on)
    labels = [-100]*len(question_ids)
    # print(len(question_ids))
    if len(question_ids) >= max_position_embeddings:
        response_length = 0
    else:
       response_length = max_position_embeddings-len(question_ids)
       response_ids = tokenizer(example['response'], truncation=True, padding='max_length', max_length=response_length)['input_ids']
       labels += response_ids

    # Add labels to inputs
    inputs['labels'] = labels

    # if len(inputs['labels']) > max_position_embeddings:
    #     print("ERROR!!!!!!!!!!!!!")

    return inputs



train_dataset = train_dataset.map(tokenize_function, batched=False)
print('Done tokenizing dataset')