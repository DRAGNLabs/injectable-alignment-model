# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

''' 
TODO: Trainer: 
    'forward' func for training

Research eventually: 
        - RMSNorm
        - rotary embeddings
        - Fairscale library (how does it parallelize, etc.)
        - Does Llama-2 parallelize with DP, TP, PP, ZeRO
'''


import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict, Literal
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from memory_utils import MemoryTrace
import numpy as np
import pandas as pd
from llama_model import ModelArgs, Transformer
from llama_tokenizer import Tokenizer
from sklearn.model_selection import train_test_split  # TODO: different function for test + train + validate?
from llama_config import train_config
from tqdm import tqdm

Role = Literal["system", "user", "assistant"]

# TypedDict = Tuple
# Literal = Tuple

class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

class Rocket_DataSet(torch.utils.data.Dataset):  # our data loader

    def __init__(self, path_to_data, pad_tok=-1, bos_tok=1, eos_tok=2, sequence_length=1001):
        self.df:pd.DataFrame = pd.read_pickle(path_to_data)
        self.train, self.test = train_test_split(self.df, test_size=.2)
        self.pad_tok = pad_tok
        self.bos_tok = bos_tok
        self.eos_tok = eos_tok
        self.sequence_length = sequence_length

        # self.get_lengths()


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
        
        
class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        dataset_path: str,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
    ) -> "Llama":
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        if len(checkpoints) > 0:
            assert model_parallel_size == len(checkpoints), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
            ckpt_path = checkpoints[get_model_parallel_rank()]
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            with open(Path(ckpt_dir) / "params.json", "r") as f:
                params = json.loads(f.read())
        else:
            params = {}

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        if len(checkpoints) > 0:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        dataset = Rocket_DataSet(dataset_path)
        return Llama(model, tokenizer, dataset)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, dataset:Rocket_DataSet):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset:Rocket_DataSet = dataset

    # @torch.inference_mode(but_training_tho) # This might be important
    # def train_by_us(self, batch_size:int, epochs=2):
    #     self.model.train()  # move to 'train' mode, as opposed to inference
    
    #     # Define tools
    #     dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    #     optim = torch.optim.Adam(self.model.parameters(), lr=0.0001)  # model.paramaters = weights tensor
    #     criterion = torch.nn.CrossEntropyLoss()

    #     for epoch in range(epochs):  # for each epoch,
            # for batch, (x,y) in enumerate(dataloader):  # look at all the data
            #     # print(f"\nBatch:\n {batch}, \nX:\n {x}, \nY:\n{y}")
            #     optim.zero_grad()  # zero out gradients if you don't want to accumulate
            #     logits = self.model.forward(tokens=x, start_pos=0)  # TODO: Figure out start position (attention caching?)
            #     loss = criterion(input=logits, target=y, ignore_index=-1)
            #     logits.backward()  # Back prop; PyTorch call it on the tensor, not the model ‾\(._.)/‾
            #     optim.step()  # apply the calculated 
    
    def train_llama_wrapper(self, batch_size):
        dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle=True, generator=torch.Generator(device=device))
        optim = torch.optim.Adam(self.model.parameters(), lr=0.0001)  # model.paramaters = weights tensor
        criterion = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 1)  # We'll probably want to change this
        trn_config=train_config()
        self.train(model=self.model, train_dataloader=dataloader, eval_dataloader=None, tokenizer=self.tokenizer, optimizer=optim, criterion=criterion, lr_scheduler=lr_scheduler, gradient_accumulation_steps=1, train_config=trn_config)


    def train(self, model, train_dataloader, eval_dataloader, tokenizer, optimizer, criterion, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None):
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
        if train_config.enable_fsdp:
            world_size = int(os.environ["WORLD_SIZE"]) 
        
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
                    if train_config.enable_fsdp:
                        x = x.to(local_rank)
                        y_true = y_true.to(local_rank)
                    else:
                        x = x.to(device)
                        y_true = y_true.to(device)
                    # loss = model(**batch).loss
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
            if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            train_epoch_loss = total_loss / len(train_dataloader)
            if train_config.enable_fsdp:
                train_epoch_loss = train_epoch_loss/world_size
            train_perplexity = torch.exp(train_epoch_loss)
            
            train_prep.append(train_perplexity)
            train_loss.append(train_epoch_loss)
            
            if train_config.enable_fsdp:
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
        if train_config.enable_fsdp and not train_config.use_peft:
            save_train_params(train_config, fsdp_config, rank)
            
        return results



        
        
        
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]


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
    path_to_dataset = "/home/dsg2060/Rocket/rocket_test/Training/tokenized_files/toy_tokenized_data.pkl"
    ckpt_dir = ""
    tokenizer_path = "../tokenizer.model"
    max_seq_len = 512
    max_batch_size = 8

    Drew_and_Jays_Llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        dataset_path=path_to_dataset
        )
    
    Drew_and_Jays_Llama.train_llama_wrapper(batch_size=10)

    print('\nNo errors!\n')

if __name__ == "__main__":
    main()