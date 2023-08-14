from transformers import AutoTokenizer

checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)