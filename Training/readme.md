Follow instructions here https://github.com/facebookresearch/llama
Request token from https://ai.meta.com/resources/models-and-libraries/llama-downloads/
./download.sh
Paste the url sent to your email
Move tokenizer.model into top level of the repo (or whatever you are using)
Move dataset file(s) into /Training/dataset/tokenized_files
run add_tokens after putting the string '<pad>' into the special_tokens list
