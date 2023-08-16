from transformers import AutoTokenizer

path = "../../test_llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(path)