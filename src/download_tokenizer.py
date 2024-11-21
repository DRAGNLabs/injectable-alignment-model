
from transformers import AutoTokenizer
from transformers import LlamaTokenizer as HFTokenizer

def download_tokenizer(model_path = "meta-llama/Llama-2-7b-chat-hf", token = ""):
    print("instantiating pretrained tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained("./local_hf_tokenizer")


download_tokenizer()