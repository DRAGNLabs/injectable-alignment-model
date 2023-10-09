# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch
from torch.utils.data import DataLoader
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import torch.nn.functional as F

import time
import glob
import json
import os
import pandas as pd
from pathlib import Path
from typing import List
from memory_utils import MemoryTrace
from sklearn.model_selection import train_test_split  # TODO: different function for test + train + validate?
from tqdm import tqdm
import yaml

from tokenizer.llama_tokenizer import Tokenizer
from llama_config import train_config
from llama_model import Transformer
from contextlib import nullcontext

from checkpoint_utils import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class Rocket_DataSet(torch.utils.data.Dataset):
    def __init__(self, path_to_data, pad_tok, bos_tok, eos_tok, sequence_length):
        self.df:pd.DataFrame = pd.read_pickle(path_to_data)
        self.train, self.test = train_test_split(self.df, test_size=.2)
        self.pad_tok = pad_tok
        self.bos_tok = bos_tok
        self.eos_tok = eos_tok
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.train.index)
    
    def __getitem__(self, index):
        pd_series_item = self.train.iloc[index,:]  # Returns a pd.Series
        tensor_item:List[int] = pd_series_item.iloc[1]  # Grab text from series
        # print(type(tensor_item), tensor_item)

        # Handle Padding
        if len(tensor_item) < self.sequence_length:
            n:int = self.sequence_length - len(tensor_item)
            pads:List[int] = [self.pad_tok]*n
            tensor_item:List[int] = tensor_item + pads

        return (torch.tensor(tensor_item[:self.sequence_length-1]).to(device), torch.tensor(tensor_item[1:self.sequence_length]).to(device))  # handles truncation 

class LLaMA:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        dataset_path: str,
        max_batch_size: int,
    ) -> "LLaMA":
     
        start_time = time.time()
        # TODO: need to test/implement this
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        if len(checkpoints) > 0:
            checkpoint = torch.load(ckpt_dir, map_location="cpu")
            with open(Path(ckpt_dir) / "params.json", "r") as f:
                params = json.loads(f.read())
        else:
            params = {}

        train_args = train_config(max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params)
        
        tokenizer = Tokenizer(model_path=tokenizer_path)  # including this for the special tokens (i.e. pad)

        train_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)  # for using GPU

        model = Transformer(train_args)

        if len(checkpoints) > 0:
            model.load_state_dict(checkpoint, strict=False)

        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        dataset = Rocket_DataSet(dataset_path, pad_tok=tokenizer.pad_id, bos_tok=tokenizer.bos_id, eos_tok=tokenizer.eos_id, sequence_length=train_args.max_seq_len)
        return LLaMA(model, tokenizer, dataset, train_args)

    def __init__(self, 
                 model: Transformer, 
                 tokenizer: Tokenizer, 
                 dataset:Rocket_DataSet,
                 train_args: train_config):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.train_args = train_args

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
    
    # TODO: change all the stuff being passed in that's part of the class
    def train_llama_wrapper(self):
        dataloader = DataLoader(self.dataset, batch_size = self.train_args.max_batch_size, shuffle=True, generator=torch.Generator(device=device))
        eval_dataloader = DataLoader(self.dataset, batch_size = self.train_args.max_batch_size, shuffle=True, generator=torch.Generator(device=device))
        optim = torch.optim.Adam(self.model.parameters(), lr=self.train_args.lr)  # model.paramaters = weights tensor
        criterion = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 1)  # We'll probably want to change this

        # TODO: This returns a results data structure containing metrics, do something with it
        self.train(model=self.model, train_dataloader=dataloader, eval_dataloader=eval_dataloader, optimizer=optim, criterion=criterion, lr_scheduler=lr_scheduler, gradient_accumulation_steps=4, fsdp_config=None, local_rank=None, rank=None)

    # TODO: you don't need to pass model in, it's a class member
    def train(self, model, train_dataloader, eval_dataloader, optimizer, criterion, lr_scheduler, gradient_accumulation_steps, fsdp_config=None, local_rank=None, rank=None):
        """
        Trains the model on the given dataloader
    
        Args:
            model: The model to be trained
            train_dataloader: The dataloader containing the training data
            optimizer: The optimizer used for training
            lr_scheduler: The learning rate scheduler
            gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
            num_epochs: The number of epochs to train for
            local_rank: The rank of the current node in a distributed setting
            eval_dataloader: The dataloader containing the eval data
        
        Returns: results dictionary containing average training and validation perplexity and loss
        """
        # Create a gradient scaler for fp16
        if self.train_args.use_fp16 and self.train_args.enable_fsdp:
            scaler = ShardedGradScaler()
        elif self.train_args.use_fp16 and not self.train_args.enable_fsdp:
            scaler = torch.cuda.amp.GradScaler()
        if self.train_args.enable_fsdp:
            world_size = int(os.environ["WORLD_SIZE"])
        autocast = torch.cuda.amp.autocast if self.train_args.use_fp16 else nullcontext
        
        train_prep = []
        train_loss = []
        val_prep = []
        val_loss =[]
        epoch_times = []
        checkpoint_times = []
        results = {}
        best_val_loss = float("inf")

        for epoch in range(self.train_args.num_epochs):
            epoch_start_time = time.perf_counter()
            with MemoryTrace() as memtrace:  # track the memory usage
                model.train()
                total_loss = 0.0
                total_length = len(train_dataloader)//gradient_accumulation_steps
                pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length)
                for step, batch in enumerate(train_dataloader): # batch = (x, y_true)
                    for element in batch:
                        if self.train_args.enable_fsdp:
                            element = element.to(local_rank) # This isn't unnecessary
                        else:
                            element = element.to(device) # This is unneccessary, they are being put on device in the dataloader

                    (x, y_true) = batch

                    with autocast(): # autocast is torch package for running in mixed precision, which improves performance
                        y_hat = model(x)
                        loss = criterion(y_hat, y_true)

                    loss = loss/gradient_accumulation_steps
                    total_loss += loss.detach().float()
                    if self.train_args.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)

                    pbar.set_description(f"Training Epoch: {epoch+1}/{self.train_args.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
                pbar.close()
                    
            epoch_end_time = time.perf_counter()-epoch_start_time
            epoch_times.append(epoch_end_time)    
            # Reducing total_loss across all devices if there's more than one CUDA device
            if torch.cuda.device_count() > 1 and self.train_args.enable_fsdp:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            train_epoch_loss = total_loss / len(train_dataloader)
            if self.train_args.enable_fsdp:
                train_epoch_loss = train_epoch_loss/world_size
            train_perplexity = torch.exp(torch.tensor(train_epoch_loss))
            
            train_prep.append(train_perplexity)
            train_loss.append(train_epoch_loss)
            if self.train_args.enable_fsdp:
                if rank==0:
                    print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                    print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                    print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                    print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                    print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
            else:
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        
            # Update the learning rate as needed
            lr_scheduler.step()
            
            # TODO: All this stuff with checkpointing needs to be implemented
            if self.train_args.run_validation:
                eval_ppl, eval_epoch_loss = self.evaluation(model, criterion, eval_dataloader, local_rank)
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
                        model.save_pretrained(self.train_args.output_dir)  
                        if self.train_args.enable_fsdp:
                            if rank==0: 
                                print(f"PEFT modules are saved in {self.train_args.output_dir} directory")
                        else:
                            print(f"PEFT modules are saved in {self.train_args.output_dir} directory")
                            
                    else:
                        if not self.train_args.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                            
                            save_model_checkpoint(
                                model, optimizer, rank, self.train_args, epoch=epoch
                            )
                        elif not self.train_args.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                            print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                            print("=====================================================")
                            
                            save_model_and_optimizer_sharded(model, rank, self.train_args)
                            if self.train_args.save_optimizer:
                                save_model_and_optimizer_sharded(model, rank, self.train_args, optim=optimizer)
                                print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                print("=====================================================")

                        if not self.train_args.use_peft and  self.train_args.save_optimizer:
                            save_optimizer_checkpoint(
                                model, optimizer, rank, self.train_args, epoch=epoch
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
        if self.train_args.enable_fsdp and not self.train_args.use_peft:
            save_train_params(self.train_args, fsdp_config, rank)
        
        #saving the training params including fsdp setting for reference.
        return results

    def evaluation(self, model, criterion, eval_dataloader, local_rank):
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
        model.eval()
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
                    y_hat = model(x)
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

# TODO: this needs to be implemented/tested
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
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        num_input_tokens = [len(t) for t in prompt_tokens]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
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
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            if self._should_stop(tokens, prompt_tokens, stop_ids, stop_words):
                break
        
        tokens[tokens == self.tokenizer.pad_id] = self.tokenizer.eos_id
        decoded = []
        num_generated_tokens = []
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


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

# TODO: This is only being called if FSDP is enabled..why? Maybe split it up, seems useful.
def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries, 
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

def main():
    print(device, '\n')
    path_to_dataset = "../tokenizer/tokenized_files/toy_tokenized_data.pkl"
    ckpt_dir = ""
    tokenizer_path = "../../tokenizer.model"
    max_seq_len = 512
    #TODO: Check batch size
    max_batch_size = 1

    Drew_and_Jay_and_Jacksons_Llama = LLaMA.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        dataset_path=path_to_dataset,
        )
    
    Drew_and_Jay_and_Jacksons_Llama.train_llama_wrapper()

    print('\nNo errors!\n')

if __name__ == "__main__":
    main()
