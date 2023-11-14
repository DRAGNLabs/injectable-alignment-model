import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from typing import List

from tokenizer.tokenizer import Tokenizer
from model import Transformer

torch.set_float32_matmul_precision('medium')

class LLaMA(LightningModule):
    def __init__(self,
                 tokenizer: Tokenizer, 
                 config: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.model = Transformer(config)

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        # TODO: verify it's training on all of the data?
        (x, y_true) = batch
        #with autocast(): # autocast is torch package for running in mixed precision, which improves performance
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y_true)

        loss = loss/self.config.gradient_accumulation_steps

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y_true) = batch
        y_hat = self.model(x)
        eval_loss = F.cross_entropy(y_hat, y_true)

        # Decode predictions and add to evaluation predictions list
        preds = torch.argmax(y_hat, 1).detach().cpu().tolist()

        # Uncomment to view predictions throughout training
        #print('preds: ', preds)
        #print('preds shape: ', len(preds))
        decoded = self.tokenizer.decode(preds)

        #eval_epoch_loss = eval_loss / len(eval_dataloader)
        #eval_ppl = torch.exp(eval_epoch_loss)
        self.log('val_loss', eval_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # TODO: log perplexity?
        
        return eval_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)  # model.paramaters = weights tensor
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer], [lr_scheduler]

    """def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_ids: List[int] = None,
        stop_words: List[str] = None,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        bsz = len(prompts)
        assert bsz <= self.train_args.batch_size, (bsz, self.train_args.batch_size)

        # Encode all prompts
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        num_input_tokens = [len(t) for t in prompt_tokens]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        # This is how far it will generate, including prompts tokens
        total_len = min(self.train_args.seq_len, max_gen_len + max_prompt_size)

        # Initialize (bsz, total_len) sized tensor with padding tokens
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        # For each prompt, input into tokens matrix
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()

        # Creates a mask where every position that is a padding token is false
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        
        # For positions in range start_pos(position after prompt) to total_len(prompt length + max generation length)
        for cur_pos in range(start_pos, total_len):
            #logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            # Logits is of shape [bsz, vocab_size, sequence_length]. Here, we grab the last token in the sequence to process only it's probabilities.
            logits = self.model(tokens[:, prev_pos:cur_pos])[:, :, -1] #TODO: not having prev_pos for attention may cause problems in this generation script, may have to rework
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
                next_token = self.sample_top_p(probs, top_p) # shape [1,1], bsz, pred
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
            if self._should_stop(tokens, prompt_tokens, stop_ids, stop_words):
                break
        
        # Turn all padding tokens into eos tokens
        tokens[tokens == self.tokenizer.pad_id] = self.tokenizer.eos_id
        decoded = []
        num_generated_tokens = []

        # Decode all generated tokens
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                num_generated_tokens.append(t.index(self.tokenizer.eos_id) - len(prompt_tokens[i]))
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                num_generated_tokens.append(max_gen_len)
            decoded.append(self.tokenizer.decode(t))
        return decoded, dict(num_input_tokens=num_input_tokens, num_generated_tokens=num_generated_tokens)
    
    def _should_stop(self, tokens, prompt_tokens, stop_ids, stop_words):
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
                g[g == self.tokenizer.pad_id] = self.tokenizer.eos_id
                g = g.tolist()
                d = self.tokenizer.decode(g)
                for stop_word in stop_words:
                    if stop_word in d:
                        do_stop[i] = True

            if all(do_stop):
                return True

        return False
    
    def sample_top_p(self, probs, p):
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
        return next_token"""