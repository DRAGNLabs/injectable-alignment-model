This repository was originally intended to be used with the Llama 2 architecture. These are Llama 2 specific instructions.

#### Using Original Llama 2 Tokenizer

To obtain the original Llama 2 Tokenizer, [Request access for Llama 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

Clone the [repository](https://github.com/facebookresearch/llama).

When download link has been obtained via email, run `./download.sh` in the llama repo.

When asked, paste the url sent to your email.

Once downloaded, move tokenizer.model into Tokenizers folder of Rocket repo.

Move dataset file(s) into `/Dataset/raw`

The tokenizer being used utilizes sentencepiece. By default, sentencepiece uses -1 as the id for padding tokens, meaning padding is disabled by default. This causes problems if you want to use a padding token. To add a new token representing padding, you can run [add_tokens.py](./tokenizer/add_tokens.py) after putting the string `<pad>` into the special_tokens list; this should already be present. Additionally, you will need to specify the path to the tokenzier within this script. The new tokenizer will have the additional padding token. Then, in [tokenizer.py](./tokenizer/tokenizer.py), ensure that `pad_id` in the tokenizer class is set to the string you defined for padding, rather than the SentencePieceProcessor `pad_id`.