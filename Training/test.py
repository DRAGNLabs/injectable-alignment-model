from transformers import AutoTokenizer

checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

print(len(tokenizer))

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_special_tokens({'sep_token': '<SEP>'})

print(len(tokenizer))