import argparse
from Rocket.rocket_test.Training.HF_Scripts.model import model, tokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

def load_datasets(data_files):
    raw_datasets = load_dataset("parquet", data_files=data_files)
    return raw_datasets['train']

def tokenize_example(example):
    max_position_embeddings = 2048  # Max context length that this model will ever be used with-- can go up to 4096.

    prompt = example['system_prompt'] + '<SEP>' + example['question'] + '<SEP>'
    inputs = tokenizer(prompt, example['response'], truncation=True, padding='max_length', max_length=max_position_embeddings)

    question_ids = tokenizer(prompt, truncation=True, max_length=max_position_embeddings)['input_ids']

    labels = [-100] * len(question_ids)

    if len(question_ids) < max_position_embeddings:
        response_length = max_position_embeddings - len(question_ids)
        response_ids = tokenizer(example['response'], truncation=True, padding='max_length', max_length=response_length)['input_ids']
        labels += response_ids

    inputs['labels'] = labels

    return inputs

def train_model(train_dataset, data_collator, training_args, save_path):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(save_path)

def main():
    batched = False
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-t', action='store_true', help='Run only .map function')
    args = parser.parse_args()

    model_name = 'test'
    print(f'Model Name: {model_name}')
    save_path = "../Models/" + model_name

    path = 'dataset/'

    data_files = {
        'train': [
            f'{path}1M-GPT4-Augmented.parquet',
            f'{path}3_5M-GPT3_5-Augmented.parquet'
        ]
    }

    train_dataset = load_datasets(data_files)

    if args.t:
        train_dataset = train_dataset.map(tokenize_example, batched=batched)
        return

    training_args = TrainingArguments(
        learning_rate=1e-3,
        output_dir="./checkpoints",
        num_train_epochs=.1,
        per_device_train_batch_size=4,
        warmup_steps=500,
        logging_dir="./logs"
    )

    train_dataset = train_dataset.map(tokenize_example, batched=batched)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_model(train_dataset, data_collator, training_args, save_path)

    prompt = "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.<SEP>You will be given a definition of a task first, then some input of the task.\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\n\nAFC Ajax (amateurs)'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\nOutput:<SEP>"
    encoded = tokenizer.encode(prompt, return_tensors="pt")

    # I think this might not work?
    model.to('cpu')

    output_answer = model.generate(encoded, max_new_tokens=200)
    decoded_answer = tokenizer.decode(output_answer[0])
    print("Generated Answer:", decoded_answer)

if __name__ == '__main__':
    main()