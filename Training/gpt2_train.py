import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

# Load the text dataset from the datasets library
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Specify the fraction of data to use for training
train_fraction = 0.001  # You can adjust this fraction

# Calculate the number of samples for training
num_train_samples = int(len(dataset["train"]) * train_fraction)
train_dataset = dataset["train"].select(range(num_train_samples))

# Initialize the GPT-2 model and tokenizer
model_name = "gpt2-medium"  # You can choose different model sizes, e.g., "gpt2", "gpt2-medium", "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize and preprocess the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

# Prepare the DataLoader for training
train_dataloader = torch.utils.data.DataLoader(tokenized_train_dataset, batch_size=4, shuffle=True)

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * 5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(5):  # You can adjust the number of epochs
    for batch in train_dataloader:
        inputs = torch.stack([item["input_ids"] for item in batch]).to(device)
        labels = torch.stack([item["input_ids"] for item in batch]).to(device)  # GPT-2 is autoregressive, so labels are the same as inputs

        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch + 1}, Batch loss: {loss.item()}")


# Save the trained model
model.save_pretrained("trained_gpt2_model")
