import sys
import yaml
import os
import torch
from typing import List

from llama import LLaMA
from tokenizer.tokenizer import Tokenizer
from utils.data_utils import Struct

device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

def generate(
    model,
    tokenizer,
    prompts: List[str], 
    #dataset: DataSet, #TODO: could be nice to use a dataloader, but this function generates for all prompts simultaneously, so it would have to be modified
    sequence_length: int,
    batch_size: int,
    max_gen_len: int,
    temperature: float = 0.8,
    top_p: float = 0.95,
    stop_ids: List[int] = None,
    stop_words: List[str] = None,
    repetition_penalty: float = 1.0,
) -> List[str]:
    bsz = len(prompts)
    assert bsz <= batch_size, (bsz, batch_size)

    # Encode all prompts
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    num_input_tokens = [len(t) for t in prompt_tokens]

    min_prompt_size = min([len(t) for t in prompt_tokens])
    max_prompt_size = max([len(t) for t in prompt_tokens])

    # This is how far it will generate, including prompts tokens
    total_len = min(sequence_length, max_gen_len + max_prompt_size)

    # Initialize (bsz, total_len) sized tensor with padding tokens
    tokens = torch.full((bsz, total_len), tokenizer.pad_id).cuda().long()

    # For each prompt, input into tokens matrix
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t).long()

    # Creates a mask where every position that is a padding token is false
    input_text_mask = tokens != tokenizer.pad_id
    start_pos = min_prompt_size
    prev_pos = 0
    
    # For positions in range start_pos(position after prompt) to total_len(prompt length + max generation length)
    for cur_pos in range(start_pos, total_len):
        #logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        # Logits is of shape [bsz, vocab_size, sequence_length]. Here, we grab the last token in the sequence to process only it's probabilities.
        input = tokens[:, prev_pos:cur_pos].to(device)
        logits = model(input)[:, :, -1] #TODO: not having prev_pos for attention may cause problems in this generation script, may have to rework
        if repetition_penalty != 1.0:
            logits_new = logits.clone()
            batch_size = len(tokens)
            for i in range(batch_size):
                for token in set(tokens[i].tolist()):
                    if logits[i, token] < 0:
                        logits_new[i, token] = logits[i, token] * repetition_penalty
                    else:
                        logits_new[i, token] = logits[i, token] / repetition_penalty
            logits = logits_new
        if temperature > 0:
            # Take softmax on logits/temperature, which evens out the probabilities, allowing more variation
            probs = torch.softmax(logits / temperature, dim=1)
            # Sample
            next_token = sample_top_p(probs, top_p) # shape [1,1], bsz, pred
        else:
            # Just grab top logit
            next_token = torch.argmax(logits, dim=1)

        # Reshape to simplify tensor; remove unnecessary dimensions basically.
        next_token = next_token.reshape(-1)

        # if input_text_mask at the cur_pos is true, then next_token is tokens at that position.
        # Or, if cur_pos is part of the original prompt, then next token is whatever token is in the prompt. Otherwise, it's the prediction.
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )

        # Put predicition into tokens
        tokens[:, cur_pos] = next_token

        # Rather than updating the start position, which is what was done for grouped attention, we pass in the entire sequence each time
        #prev_pos = cur_pos

        # Check if generation should be stopped (if a stop token was generated, for example)
        if _should_stop(tokenizer, tokens, prompt_tokens, stop_ids, stop_words):
            break
    
    # Turn all padding tokens into eos tokens
    tokens[tokens == tokenizer.pad_id] = tokenizer.eos_id
    decoded = []
    num_generated_tokens = []

    # Decode all generated tokens
    for i, t in enumerate(tokens.tolist()):
        # cut to max gen len
        t = t[: len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        try:
            num_generated_tokens.append(t.index(tokenizer.eos_id) - len(prompt_tokens[i]))
            t = t[: t.index(tokenizer.eos_id)]
        except ValueError:
            num_generated_tokens.append(max_gen_len)
        decoded.append(tokenizer.decode(t))
    return decoded, dict(num_input_tokens=num_input_tokens, num_generated_tokens=num_generated_tokens)

def _should_stop(tokenizer, tokens, prompt_tokens, stop_ids, stop_words):
    if stop_ids is not None:
        do_stop = [False for _ in range(len(tokens))]
        for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
            g = t[len(p):].tolist()
            for stop_id in stop_ids:
                if stop_id in g:
                    do_stop[i] = True

        if all(do_stop):
            return True

    if stop_words is not None:
        do_stop = [False for _ in range(len(tokens))]
        for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
            t = t.clone()
            g = t[len(p):]
            g[g == tokenizer.pad_id] = tokenizer.eos_id
            g = g.tolist()
            d = tokenizer.decode(g)
            for stop_word in stop_words:
                if stop_word in d:
                    do_stop[i] = True

        if all(do_stop):
            return True

    return False

def sample_top_p(probs, p):
    # sort probs in ascending order
    probs_sort, probs_idx = torch.sort(probs, dim=1, descending=True) # NOTE: I changed dim from -1 to 1
    probs_sum = torch.cumsum(probs_sort, dim=1)
    # Mask out values below p in the cumulative sum
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    # Divide each element by the sum
    probs_sort.div_(probs_sort.sum(dim=1, keepdim=True))
    # Sample once from probability dist
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Pull token out from probs_idx
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def inference(config):
    tokenizer = Tokenizer(model_path=config.tokenizer_path)  # including this for the special tokens (i.e. pad)
    config.vocab_size = tokenizer.n_words
    config.pad_id = tokenizer.pad_id

    # Build model class
    model = LLaMA(tokenizer=tokenizer, config=config)

    # Load checkpoint
    checkpoint_path=config.checkpoint_path

    print(f"Using checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])

    model.cuda()
    model.eval()
    
    #dm = DataModule(config.train_path, config.eval_path, tokenizer, config.batch_size, config.sequence_length)
    # Load dataset for inference, create dataloader
    # TODO: implement the dataloader stuff, instead of just a list of strings
    #inference_dataset_path = config.inference_dataset_path
    #inference_dataset = DataSet(inference_dataset_path,
    #                                    tokenizer.pad_id, 
    #                                    tokenizer.bos_id, 
    #                                    tokenizer.eos_id, 
    #                                    config.sequence_length)

    # Generate
    #prompt = ["You are an AI assistant. You will be given a task. You must generate a detailed and long answer.	Generate an approximately fifteen-word sentence that describes all this data: Midsummer House eatType restaurant; Midsummer House food Chinese; Midsummer House priceRange moderate; Midsummer House customer rating 3 out of 5; Midsummer House near All Bar One"] # Load data here
    with open(config.inference_dataset_path, 'r') as f:
        prompts = f.readlines()

    decoded, dictionary = generate(model,
                                   tokenizer,
                                   prompts,
                                   sequence_length=config.sequence_length,
                                   batch_size=config.batch_size,
                                   max_gen_len = config.max_gen_len,
                                   repetition_penalty=9.0)

    print('decoded: ', decoded)
    print('dictionary: ', dictionary)

    print('\nNo errors!\n')

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    inference(config)

if __name__ == "__main__":
    main()