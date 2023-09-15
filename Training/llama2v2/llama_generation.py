# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

device = 'cpu'

import torch
torch.set_default_device('cpu')
from torch.utils.data import DataLoader
import time
import glob
import pandas as pd
import matplotlib as plt
from llama_model import ModelArgs
from Rocket.rocket_test.Training.llama2v2.tokenizer.llama_tokenizer import Tokenizer
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict, Literal
import torch.nn.functional as F
from memory_utils import MemoryTrace
import numpy as np
from sklearn.model_selection import train_test_split  # TODO: different function for test + train + validate?
from llama_config import train_config
from tqdm import tqdm
from llama_model import Transformer

class Rocket_DataSet(torch.utils.data.Dataset):  # our data loader

    def __init__(self, path_to_data, pad_tok=-1, bos_tok=1, eos_tok=2, sequence_length=1001):
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
    
    def get_lengths(self, filter=True, min_length=0, max_length=2000, make_plot=True):
        """
        Get the lengths of sequences

        filter: Bool, makes 
        """
        # range, mean, std dev?
        seq_lengths:pd.Series = self.train.apply(lambda x: len(x[1]), axis=1)

        # Filter the Series to include only sequences within the specified range
        if filter==True:
            seq_lengths = seq_lengths[(seq_lengths >= min_length) & (seq_lengths <= max_length)]
        
        if make_plot==False:
            return seq_lengths

        # Create a histogram
        plt.figure(figsize=(10, 6))
        plt.hist(seq_lengths, bins=100, color='skyblue', edgecolor='black')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        plt.title('Sequence Length Distribution (Histogram)')

        # Calculate the range
        data_range = seq_lengths.max() - seq_lengths.min()

        # Calculate the mean
        mean_value = round(seq_lengths.mean())

        # Calculate the median
        median_value = seq_lengths.median()

        # Calculate the standard deviation
        std_deviation = round(seq_lengths.std())

        # Calculate the Interquartile Range (IQR)
        Q1 = seq_lengths.quantile(0.25)
        Q3 = seq_lengths.quantile(0.75)
        IQR = Q3 - Q1

        # Define a threshold for identifying outliers (e.g., 1.5 times the IQR)
        outlier_threshold = 1.5 * IQR

        # Identify outliers
        outliers = seq_lengths[
            (seq_lengths < (Q1 - outlier_threshold)) | 
            (seq_lengths > (Q3 + outlier_threshold))
        ]

        # Print the results
        # print(f"Range: {data_range}")
        # print(f"Mean: {mean_value}")
        # print(f"Median: {median_value}")
        # print(f"Standard Deviation: {std_deviation}")
        # print("Outliers:")
        # print(outliers)
        plt.text(0.79, 0.9, f"Range: {data_range}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.79, 0.85, f"Mean: {mean_value}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.79, 0.8, f"Std Dev: {std_deviation}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.79, 0.75, f"Median: {median_value}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.79, 0.7, f"Q1: {Q1}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.79, 0.65, f"Q3: {Q3}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.79, 0.6, f"Outliers: {len(outliers)}", transform=plt.gca().transAxes, fontsize=12)

        # Add dotted lines at standard deviation intervals from the mean
        plt.axvline(x=mean_value, color='orange', linestyle='--', label='Mean')
        plt.axvline(x=mean_value - std_deviation, color='gray', linestyle='--', label='-1 Std Dev')
        plt.axvline(x=mean_value + std_deviation, color='gray', linestyle='--', label='+1 Std Dev')
        plt.axvline(x=mean_value - 2 * std_deviation, color='gray', linestyle='--', label='-2 Std Dev')
        plt.axvline(x=mean_value + 2 * std_deviation, color='gray', linestyle='--', label='+2 Std Dev')

        plt.savefig('gpt3-1_Sequence_Plot.png')

        return seq_lengths
        

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
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        if len(checkpoints) > 0:
            checkpoint = torch.load(ckpt_dir, map_location="cpu")
            with open(Path(ckpt_dir) / "params.json", "r") as f:
                params = json.loads(f.read())
        else:
            params = {}

        model_args: ModelArgs = ModelArgs()
        tokenizer = Tokenizer(model_path=tokenizer_path)  # including this for the special tokens (i.e. pad)

        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        if len(checkpoints) > 0:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        dataset = Rocket_DataSet(dataset_path)
        return LLaMA(model, tokenizer, dataset)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, dataset:Rocket_DataSet):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset:Rocket_DataSet = dataset


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
    

    def train_llama_wrapper(self, batch_size):
        dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle=True, generator=torch.Generator(device=device))
        optim = torch.optim.Adam(self.model.parameters(), lr=0.0001)  # model.paramaters = weights tensor
        criterion = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 1)  # We'll probably want to change this
        trn_config=train_config()
        self.train(model=self.model, train_dataloader=dataloader, eval_dataloader=None, tokenizer=self.tokenizer, optimizer=optim, criterion=criterion, lr_scheduler=lr_scheduler, gradient_accumulation_steps=1, train_config=trn_config)


    def train(self, model, train_dataloader, eval_dataloader, tokenizer, optimizer, criterion, lr_scheduler, gradient_accumulation_steps, train_config):
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
            train_config: The training configuration
            eval_dataloader: The dataloader containing the eval data
            tokenizer: tokenizer used in the eval for decoding the predicitons
        
        Returns: results dictionary containing average training and validation perplexity and loss
        """
        # Create a gradient scaler for fp16
        if train_config.use_fp16 and train_config.enable_fsdp:
            scaler = ShardedGradScaler()
        elif train_config.use_fp16 and not train_config.enable_fsdp:
            scaler = torch.cuda.amp.GradScaler() 
        
        train_prep = []
        train_loss = []
        val_prep = []
        val_loss =[]
        epoch_times = []
        checkpoint_times = []
        results = {}
        best_val_loss = float("inf")


        for epoch in range(train_config.num_epochs):
            epoch_start_time = time.perf_counter()
            with MemoryTrace() as memtrace:  # track the memory usage
                model.train()
                total_loss = 0.0
                total_length = len(train_dataloader)//gradient_accumulation_steps
                pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length)
                for step, (x, y_true) in enumerate(train_dataloader):
                    x = x.to(device)
                    y_true = y_true.to(device)
                    y_hat = model(x)
                    loss = criterion(y_hat, y_true)
                    loss = loss / gradient_accumulation_steps
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
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

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
                pbar.close()
                    
            epoch_end_time = time.perf_counter()-epoch_start_time
            epoch_times.append(epoch_end_time)    
            # Reducing total_loss across all devices if there's more than one CUDA device
            train_epoch_loss = total_loss / len(train_dataloader)
            train_perplexity = torch.exp(train_epoch_loss)
            
            train_prep.append(train_perplexity)
            train_loss.append(train_epoch_loss)
            
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
            
            # Update the learning rate as needed
            lr_scheduler.step()
            
            # if train_config.run_validation:
            #     eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer)
            #     checkpoint_start_time = time.perf_counter()
            #     if train_config.save_model and eval_epoch_loss < best_val_loss:
            #         if train_config.enable_fsdp:
            #             dist.barrier()
            #         if train_config.use_peft:
            #             if train_config.enable_fsdp:
            #                 if rank==0:
            #                     print(f"we are about to save the PEFT modules")
            #             else:
            #                 print(f"we are about to save the PEFT modules")
            #             model.save_pretrained(train_config.output_dir)  
            #             if train_config.enable_fsdp:
            #                 if rank==0: 
            #                     print(f"PEFT modules are saved in {train_config.output_dir} directory")
            #             else:
            #                 print(f"PEFT modules are saved in {train_config.output_dir} directory")
                            
            #         else:
            #             if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                            
            #                 save_model_checkpoint(
            #                     model, optimizer, rank, train_config, epoch=epoch
            #                 )
            #             elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
            #                 print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
            #                 print("=====================================================")
                            
            #                 save_model_and_optimizer_sharded(model, rank, train_config)
            #                 if train_config.save_optimizer:
            #                     save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
            #                     print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
            #                     print("=====================================================")

            #             if not train_config.use_peft and  train_config.save_optimizer:
            #                 save_optimizer_checkpoint(
            #                     model, optimizer, rank, train_config, epoch=epoch
            #                 )
            #                 print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
            #                 print("=====================================================")                     
            #         if train_config.enable_fsdp:
            #             dist.barrier()
            #     checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            #     checkpoint_times.append(checkpoint_end_time)
            #     if eval_epoch_loss < best_val_loss:
            #         best_val_loss = eval_epoch_loss
            #         if train_config.enable_fsdp:
            #             if rank==0:
            #                 print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            #         else:
            #             print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            #     val_loss.append(best_val_loss)
            #     val_prep.append(eval_ppl)
            # if train_config.enable_fsdp:
            #     if rank==0:
            #         print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
            # else:
            #     print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        
        avg_epoch_time = sum(epoch_times)/ len(epoch_times)
        avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
        avg_train_prep = sum(train_prep)/len(train_prep)
        avg_train_loss = sum(train_loss)/len(train_loss)
        # if train_config.run_validation:
            # avg_eval_prep = sum(val_prep)/len(val_prep) 
            # avg_eval_loss = sum(val_loss)/len(val_loss) 

        results['avg_train_prep'] = avg_train_prep
        results['avg_train_loss'] = avg_train_loss
        # if train_config.run_validation:
        #     results['avg_eval_prep'] = avg_eval_prep
        #     results['avg_eval_loss'] = avg_eval_loss
        results["avg_epoch_time"] = avg_epoch_time
        results["avg_checkpoint_time"] = avg_checkpoint_time
        
        #saving the training params including fsdp setting for reference.
        return results


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

def main():
    print(device, '\n')
    path_to_dataset = "../dataset/tokenized_files/toy_tokenized_data.pkl"
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
    #TODO: check batch size
    Drew_and_Jay_and_Jacksons_Llama.train_llama_wrapper(batch_size=1)

    print('\nNo errors!\n')

if __name__ == "__main__":
    main()
