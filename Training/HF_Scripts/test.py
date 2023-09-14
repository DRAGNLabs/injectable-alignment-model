from Rocket.rocket_test.Training.HF_Scripts.model import model, tokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

model_name = 'ten_thousand'
model_path = "../Models/" + model_name

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
train_dataset = train_dataset.select([i for i in range(10000)])

max_length = 2048

# Define the tokenize function that processes a batch of examples
def tokenize_function(example):
    # Tokenize inputs
    prompt = example['system_prompt'] + '<SEP>' + example['question'] + '<SEP>' + example['response']
    inputs = tokenizer(prompt, padding='max_length')
    if len(inputs['input_ids']) >= max_length:
        diff = len(inputs['input_ids']) - max_length
        inputs['input_ids'] = inputs['input_ids'][diff:]
        inputs['labels'] = inputs['input_ids'][diff:]
        inputs['attention_mask'] = inputs['attention_mask'][diff:]
    if len(inputs['input_ids']) > max_length: #or len(question_ids) > max_position_embeddings or len(response_ids) > max_position_embeddings:
        print('ERROR')

    return inputs



train_dataset = train_dataset.map(tokenize_function, batched=False)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#! Add evaluation

# Define the training arguments
training_args = TrainingArguments(
    learning_rate=1e-3,
    output_dir="./checkpoints",
    num_train_epochs=1,  
    per_device_train_batch_size=8,
    warmup_steps=500,
    logging_dir="./logs",
    # logging_steps=100,
    # save_strategy="epoch"
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
trainer.save_model(model_path)

# Load the trained model
# trained_model = AutoModel.from_pretrained(model_path)

# Generate answers using the current model
prompt = "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.<SEP>You will be given a definition of a task first, then some input of the task.\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\n\nAFC Ajax (amateurs)'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\nOutput:<SEP>"
encoded = tokenizer.encode(prompt, return_tensors="pt")

model = model.to('cpu')

output_answer = model.generate(encoded, max_new_tokens=200)
decoded_answer = tokenizer.decode(output_answer[0])
print("Generated Answer:", decoded_answer)
