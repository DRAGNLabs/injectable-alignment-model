from model import model
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import OpenLlamaForCausalLM


path = 'Training/dataset/'

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
# train_dataset = train_dataset.select([i for i in range(1000)])
# train_dataset = train_dataset.select([0, 1, 2, 3, 4, 5])
train_dataset = train_dataset.select([i for i in range(1000)])

checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_special_tokens({'sep_token': '<SEP>'})

def tokenize_function(example):
    # Tokenize inputs
    prompt = example['system_prompt'] + '<SEP>' + example['question'] + '<SEP>'
    inputs = tokenizer(prompt, example['response'], truncation=True, padding='max_length', max_length=1024)

    # Tokenize each subsection
    question_ids = tokenizer(prompt, truncation=True)['input_ids']
    response_ids = tokenizer(example['response'], truncation=True, padding='max_length', max_length=1024-len(question_ids))['input_ids']

    # Create labels (-100 for those not backproped on)
    labels = [-100]*len(question_ids) + response_ids

    # Add labels to inputs
    inputs['labels'] = labels

    return inputs

train_dataset = train_dataset.map(tokenize_function, batched=False)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#! Add evaluation

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch"
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("path_to_save_model")

# Load the trained model
trained_model = OpenLlamaForCausalLM.from_pretrained("path_to_save_model")

# Generate answers using the trained model
input_question = "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.<SEP>What is the answer to life, the universe, and everything?<SEP>"
input_question_encoded = tokenizer.encode(input_question, return_tensors="pt")
output_answer = trained_model.generate(input_question_encoded)
decoded_answer = tokenizer.decode(output_answer[0])
print("Generated Answer:", decoded_answer)