# Rocket

There are separate READMEs in [Training](https://github.com/DRAGNLabs/Rocket/blob/main/Training/readme.md) and [Evaluation](https://github.com/DRAGNLabs/Rocket/blob/main/Evaluation/README_Eval.md) for their respective setups.

# TODO: combine these?
Follow instructions here https://github.com/facebookresearch/llama
Request token from https://ai.meta.com/resources/models-and-libraries/llama-downloads/
./download.sh
Paste the url sent to your email
Move tokenizer.model into top level of the repo (or whatever you are using)
Move dataset file(s) into /Training/dataset/tokenized_files
run add_tokens after putting the string '<pad>' into the special_tokens list
