# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger

import time
import json
import os
import pandas as pd
from pathlib import Path
from typing import List
from utils.memory_utils import MemoryTrace

from tqdm import tqdm
import yaml

from tokenizer.tokenizer import Tokenizer
from model import Transformer
from dataset import Rocket_DataSet
from contextlib import nullcontext

from utils.checkpoint_utils import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class LLaMA(LightningModule):
    def __init__(self,
                 tokenizer: Tokenizer, 
                 config: dict):
        self.tokenizer = tokenizer
        self.config = config

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        (x, y_true) = batch
        #with autocast(): # autocast is torch package for running in mixed precision, which improves performance
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y_true)

        loss = loss/self.train_args.gradient_accumulation_steps

        # TODO: log here
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)  # model.paramaters = weights tensor
        return optimizer
    


    def train(self, fsdp_config=None, local_rank=None, rank=None):
        """
        Trains the model on the given dataloader
    
        Args:
            train_dataloader: The dataloader containing the training data
            eval_dataloader: The dataloader containing the eval data
            optimizer: The optimizer used for training
            lr_scheduler: The learning rate scheduler
            local_rank: The rank of the current node in a distributed setting
        
        Returns: results dictionary containing average training and validation perplexity and loss
        """
        # Create necessary training modules based on config
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.train_args.gamma)
        
        train_prep = []
        train_loss = []
        val_prep = []
        val_loss =[]
        epoch_times = []
        checkpoint_times = []
        results = {}
        best_val_loss = float("inf")
        # Update the learning rate as needed
        lr_scheduler.step()

        # TODO: finish deconstructing this function
    """        
            if self.train_args.run_validation:
                eval_ppl, eval_epoch_loss = self.evaluation(criterion, eval_dataloader, local_rank)
                checkpoint_start_time = time.perf_counter()
                if self.train_args.save_model and eval_epoch_loss < best_val_loss:
                    if self.train_args.enable_fsdp:
                        dist.barrier()
                    if self.train_args.use_peft:
                        if self.train_args.enable_fsdp:
                            if rank==0:
                                print(f"we are about to save the PEFT modules")
                        else:
                            print(f"we are about to save the PEFT modules")
                        self.model.save_pretrained(self.train_args.output_dir)  
                        if self.train_args.enable_fsdp:
                            if rank==0: 
                                print(f"PEFT modules are saved in {self.train_args.output_dir} directory")
                        else:
                            print(f"PEFT modules are saved in {self.train_args.output_dir} directory")
                            
                    else:
                        if not self.train_args.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                            
                            save_model_checkpoint(
                                self.model, optimizer, rank, self.train_args, epoch=epoch
                            )
                        elif not self.train_args.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                            print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                            print("=====================================================")
                            
                            save_model_and_optimizer_sharded(self.model, rank, self.train_args)
                            if self.train_args.save_optimizer:
                                save_model_and_optimizer_sharded(self.model, rank, self.train_args, optim=optimizer)
                                print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                print("=====================================================")

                        if not self.train_args.use_peft and  self.train_args.save_optimizer:
                            save_optimizer_checkpoint(
                                self.model, optimizer, rank, self.train_args, epoch=epoch
                            )
                            print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                            print("=====================================================")                     
                    if self.train_args.enable_fsdp:
                        dist.barrier()
                checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                checkpoint_times.append(checkpoint_end_time)
                if eval_epoch_loss < best_val_loss:
                    best_val_loss = eval_epoch_loss
                    if self.train_args.enable_fsdp:
                        if rank==0:
                            print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                    else:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                val_loss.append(best_val_loss)
                val_prep.append(eval_ppl)
            if self.train_args.enable_fsdp:
                if rank==0:
                    print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
            else:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        avg_epoch_time = sum(epoch_times)/ len(epoch_times)
        avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
        avg_train_prep = sum(train_prep)/len(train_prep)
        avg_train_loss = sum(train_loss)/len(train_loss)
        if self.train_args.run_validation:
            avg_eval_prep = sum(val_prep)/len(val_prep) 
            avg_eval_loss = sum(val_loss)/len(val_loss) 

        results['avg_train_prep'] = avg_train_prep
        results['avg_train_loss'] = avg_train_loss
        if self.train_args.run_validation:
            results['avg_eval_prep'] = avg_eval_prep
            results['avg_eval_loss'] = avg_eval_loss
        results["avg_epoch_time"] = avg_epoch_time
        results["avg_checkpoint_time"] = avg_checkpoint_time

        #saving the training params including fsdp setting for reference.
        # TODO: why only with enable_fsdp?
        #if self.train_args.enable_fsdp and not self.train_args.use_peft:
        #    self.save_train_params(self.train_args, fsdp_config, rank)

        #saving the training params including fsdp setting for reference.
        return results
"""
    def evaluation(self, criterion, eval_dataloader, local_rank):
        """
        Evaluates the model on the given dataloader
        
        Args:
            model: The model to evaluate
            eval_dataloader: The dataloader containing the evaluation data
            local_rank: The rank of the current node in a distributed setting
            tokenizer: The tokenizer used to decode predictions
        
        Returns: eval_ppl, eval_epoch_loss
        """
        if self.train_args.enable_fsdp:
            world_size = int(os.environ["WORLD_SIZE"]) 
        self.model.eval()
        eval_preds = []
        eval_loss = 0.0  # Initialize evaluation loss
        with MemoryTrace() as memtrace:
            for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
                for element in batch:
                    if self.train_args.enable_fsdp:
                        element = element.to(local_rank)
                    else:
                        element = element.to(device) # Might want this to be 'cuda:0'
                # Ensure no gradients are computed for this scope to save memory
                with torch.no_grad():
                    # Forward pass and compute loss
                    (x, y_true) = batch
                    y_hat = self.model(x)
                    loss = criterion(y_hat, y_true)
                    eval_loss += loss.detach().float()
                # Decode predictions and add to evaluation predictions list
                preds = torch.argmax(y_hat, 1).detach().cpu().tolist()

                # Uncomment to view predictions throughout training
                #print('preds: ', preds)
                #print('preds shape: ', len(preds))
                decoded = self.tokenizer.decode(preds)

                # Nothing is being done with this list currently
                eval_preds.extend( 
                    decoded
                )
        
        # If there's more than one CUDA device, reduce evaluation loss across all devices
        if torch.cuda.device_count() > 1 and self.train_args.enable_fsdp:
            dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        
        # Compute average loss and perplexity
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        if self.train_args.enable_fsdp:
            eval_epoch_loss = eval_epoch_loss/world_size
        eval_ppl = torch.exp(eval_epoch_loss)
        
        # Print evaluation metrics
        if self.train_args.enable_fsdp:
            if local_rank==0:
                print(f" {eval_ppl=} {eval_epoch_loss=}")
        else:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
            
        return eval_ppl, eval_epoch_loss

    def generate(
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
        return next_token

    # TODO: This is only being called if FSDP is enabled..why? Maybe split it up, seems useful.
    # TODO: Do we need this? how should we this?
    # def save_train_params(self, train_config, fsdp_config, rank):
    #     """
    #     This function saves the train_config and FSDP config into a train_params.yaml.
    #     This will be used by converter script in the inference folder to fetch the HF model name or path.
    #     It also would be hepful as a log for future references.
    #     """
    #     # Convert the train_config and fsdp_config objects to dictionaries, 
    #     # converting all values to strings to ensure they can be serialized into a YAML file
    #     train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    #     fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    #     # Merge the two dictionaries into one
    #     train_params_dict = {**train_config_dict, **fsdp_config_dict}
    #     # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    #     folder_name = (
    #     train_config.dist_checkpoint_root_folder
    #     + "/"
    #     + train_config.dist_checkpoint_folder
    #     + "-"
    #     + train_config.model_name
    #     )

    #     save_dir = Path.cwd() / folder_name
    #     # If the directory does not exist, create it
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     # Convert the dictionary to a YAML string
    #     config_yaml = yaml.dump(train_params_dict, indent=4)
    #     file_name = os.path.join(save_dir,'train_params.yaml')

    #     # Check if there's a directory with the same name as the file
    #     if os.path.isdir(file_name):
    #         print(f"Error: {file_name} is a directory, not a file.")
    #     else:
    #         # Write the YAML string to the file
    #         with open(file_name, 'w') as f:
    #             f.write(config_yaml)
    #         if rank==0:
    #             print(f"training params are saved in {file_name}")
