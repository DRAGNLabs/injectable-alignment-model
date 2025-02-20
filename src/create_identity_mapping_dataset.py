import os
import sys
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from transformers import AutoTokenizer
from llama_models.injected_llama_for_causal import LlamaForCausalLM
import yaml
from utils.data_utils import Struct
import gzip

class LlamaDatasetGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def generate_single_token_dataset(self, num_samples_per_tok=100, output_dir="data/single_token", temperature=1.0):
        """Generate dataset for single token prediction task"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Common tokens to use as prompts
        common_tokens = [
            # Articles and determiners
            "the", "a", "an", "this", "that", "these", "those",
            
            # Prepositions
            "in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
            "under", "over", "through", "between", "among", "around",
            
            # Conjunctions
            "and", "but", "or", "nor", "yet", "so", "because", "while",
            
            # Common verbs
            "is", "are", "was", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "can", "could", "will", "would", "should",
            "make", "made", "take", "took", "get", "got", "go", "went",
            
            # Common adjectives
            "good", "bad", "big", "small", "high", "low", "new", "old",
            "first", "last", "same", "different", "early", "late",
            
            # Common adverbs
            "very", "really", "just", "now", "then", "here", "there",
            "well", "often", "always", "never", "sometimes",
            
            # Pronouns
            "i", "you", "he", "she", "it", "we", "they",
            "my", "your", "his", "her", "its", "our", "their",
            
            # Question words
            "what", "when", "where", "why", "who", "how",
            
            # Numbers and quantities
            "one", "two", "three", "many", "much", "some", "any", "all",
            
            # Time-related
            "today", "yesterday", "now", "soon", "later",
            
            # Common nouns
            "time", "year", "day", "way", "thing", "man", "woman", "world",
            "life", "hand", "part", "child", "eye", "place", "work", "week",
            "case", "point", "number", "group", "fact", "idea"
        ]
        dataset = []
        vocab_size = self.tokenizer.vocab_size
        
        print(f"Generating { len(common_tokens)} samples...")
        for prompt in tqdm(common_tokens):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature scaling
                scaled_logits = next_token_logits / temperature
                next_token_probs = torch.softmax(scaled_logits, dim=-1)
                
                # Convert to numpy and reduce precision
                full_probs = next_token_probs[0].cpu().numpy().astype(np.float16)
                
                sample = {
                    "prompt": prompt,
                    "prompt_tokens": input_ids[0].tolist(),
                    "temperature": temperature,
                    "full_distribution": full_probs.tolist(),
                    "vocab_size": vocab_size
                }
                dataset.append(sample)
        
        # Save with compression
        output_file = Path(output_dir) / "single_token_full_distribution_dataset.json.gz"
        with gzip.open(output_file, 'wt', encoding='UTF-8') as f:
            json.dump(dataset, f)
        print(f"Saved full distribution dataset to {output_file}")
        
    def generate_short_sequence_dataset(self, prompts, output_dir="data/short_sequence"):
        """Generate completions for short, structured prompts"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        dataset = []
        print("Generating short sequence completions...")
        for prompt in tqdm(prompts):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=50,
                    temperature=0.0,  # Deterministic
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            sample = {
                "prompt": prompt,
                "completion": completion,
                "input_ids": input_ids.tolist(),
                "output_ids": outputs.tolist()
            }
            dataset.append(sample)
        
        # Save dataset
        output_file = Path(output_dir) / "short_sequence_dataset.json"
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved short sequence dataset to {output_file}")
    
    def generate_controlled_dataset(self, prompts, output_dir="data/controlled",
                                  temperature=0.3, samples_per_prompt=5):
        """Generate multiple completions for structured prompts"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        dataset = []
        print(f"Generating controlled completions (temp={temperature})...")
        for prompt in tqdm(prompts):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=100,
                    temperature=temperature,
                    num_return_sequences=samples_per_prompt,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            completions = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            sample = {
                "prompt": prompt,
                "temperature": temperature,
                "completions": completions,
                "input_ids": input_ids.tolist(),
                "output_ids": outputs.tolist()
            }
            dataset.append(sample)
        
        # Save dataset
        output_file = Path(output_dir) / f"controlled_dataset_t{temperature}.json"
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved controlled generation dataset to {output_file}")
    
    def generate_freeform_dataset(self, prompts, output_dir="data/freeform",
                                temperature=0.7, samples_per_prompt=10):
        """Generate diverse completions for open-ended prompts"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        dataset = []
        print(f"Generating free-form completions (temp={temperature})...")
        for prompt in tqdm(prompts):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=200,
                    temperature=temperature,
                    num_return_sequences=samples_per_prompt,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            completions = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            sample = {
                "prompt": prompt,
                "temperature": temperature,
                "top_p": 0.9,
                "completions": completions,
                "input_ids": input_ids.tolist(),
                "output_ids": outputs.tolist()
            }
            dataset.append(sample)
        
        # Save dataset
        output_file = Path(output_dir) / f"freeform_dataset_t{temperature}.json"
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved free-form generation dataset to {output_file}")

def main():
    
    # Load config and model
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = Struct(**config)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    original_model_path = "/home/huang717/DRAGN/IRM/injectable-alignment-model/default_checkpoints/Llama-2-7b-chat-hf.ckpt"
    model = LlamaForCausalLM(tokenizer, config)
    checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    # Deactivate IRM
    model.model.irm.deactivate()
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Initialize generator
    generator = LlamaDatasetGenerator(model, tokenizer)
    
    # Generate datasets
    save_dir = "../datasets/identity_mapping/single_token"
    generator.generate_single_token_dataset(
        num_samples_per_tok=100,
        output_dir=save_dir,
        temperature=1.0  # Default, no temperature scaling
    )
    
    # short_prompts = [
    #     "The capital of France is",
    #     "Two plus two equals",
    #     "The sky is blue because"
    # ]
    # generator.generate_short_sequence_dataset(short_prompts)
    
    # structured_prompts = [
    #     "Complete this sentence: The weather today is",
    #     "List three colors: 1.",
    #     "Answer yes or no: Is water wet?"
    # ]
    # generator.generate_controlled_dataset(structured_prompts)
    
    # open_prompts = [
    #     "Write a short story about",
    #     "Explain why humans dream",
    #     "Describe your ideal vacation"
    # ]
    # generator.generate_freeform_dataset(open_prompts)

if __name__ == "__main__":
    main()