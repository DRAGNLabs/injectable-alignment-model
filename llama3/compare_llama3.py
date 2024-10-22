import torch
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM,AutoConfig
from llama3_8B import Llama, Transformer, Tokenizer  # Import my implementation
import random
import numpy as np

# def get_last_layer_activations(model, input_ids):
#     """Helper function to get the last layer activations."""
#     if isinstance(model, LlamaForCausalLM):
#         # For Hugging Face model
#         outputs = model(input_ids, output_hidden_states=True)
#         return outputs.hidden_states[-1]
#     elif isinstance(model, Transformer):
#         # For my implementation
#         _, activations = model(input_ids, start_pos=0)
#         return activations[-1]
#     else:
#         raise ValueError("Unsupported model type")
def get_all_layers_activations(model, input_ids):
    """Helper function to get all layers' activations."""
    if isinstance(model, LlamaForCausalLM):
        # For Hugging Face model
        outputs = model(input_ids, output_hidden_states=True)
        return outputs.hidden_states
    elif isinstance(model, Transformer):
        # For my implementation
        _, activations = model(input_ids, start_pos=0)
        return activations
    else:
        raise ValueError("Unsupported model type")
def create_causal_mask(input_ids):
    """Create a causal mask for the input sequence."""
    seq_length = input_ids.size(1)
    # Create a lower triangular matrix
    mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool))
    # Expand dimensions to match the expected shape [batch_size, 1, seq_length, seq_length]
    return mask.unsqueeze(0).unsqueeze(1)

def generate(my_model, my_tokenizer, hf_model, hf_tokenizer,prompt):
    temperature = 0.6
    top_p = 0.8
    max_seq_len = 128
    max_gen_len = 64
    max_batch_size = 4
    # Output from my llama
    my_result = my_model.text_completion(
                prompt,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                )
    
    # Output from HF llama
     # Hugging Face model generation
    # terminators = [
    #     hf_tokenizer.eos_token_id,
    #     hf_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]
    if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_inputs = hf_tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding = True)
    hf_input_ids = hf_inputs.input_ids
    attention_mask = hf_inputs["attention_mask"]

    hf_result = hf_model.generate(
        hf_input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        # eos_token_id=terminators,
        pad_token_id=hf_tokenizer.eos_token_id
    )
    hf_result = hf_tokenizer.decode(hf_result[0], skip_special_tokens=True)
    return my_result, hf_result

def compare_all_layer_activations(input_prompt):
    # Initialize my implementation
    ckpt_dir = "/home/huang717/.llama/checkpoints/Llama-3-8B/"
    tokenizer_path = "/home/huang717/.llama/checkpoints/Llama-3-8B/tokenizer.model"
    max_seq_len = 128
    max_batch_size = 1
    my_llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=1
    )
    my_model = my_llama.model.cuda()
    my_tokenizer = my_llama.tokenizer
    my_model.return_activation = True  # Enable returning activation

    # Initialize Hugging Face implementation
    print("\nTrying to load Hugging Face model")
    hf_path = "/home/huang717/.llama/checkpoints/Llama-3-8B-HF"
    # hf_model = LlamaForCausalLM.from_pretrained(
    #     hf_path,
    #     torch_dtype=torch.bfloat16).cuda()
    hf_config = AutoConfig.from_pretrained(hf_path)
    hf_config._attn_implementation = 'sdpa'

    hf_model = AutoModelForCausalLM.from_pretrained(hf_path, config = hf_config).cuda()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_path)

    # Set both models to evaluation mode
    my_model.eval()
    hf_model.eval()

    # Prepare input
    prompt = input_prompt
    my_input_ids = torch.tensor([my_tokenizer.encode(prompt, bos=True, eos=False)]).cuda()
    hf_input_ids = hf_tokenizer.encode(prompt, return_tensors="pt").cuda()
    print(f"\nCheck the input tokens the same -- {my_input_ids==hf_input_ids}")

    # Get activations for all layers
    print("\nTrying to get activations for all layers")

    start_time = time.time()
    with torch.no_grad():
        my_activations_list = get_all_layers_activations(my_model, my_input_ids)
        hf_activations_list = get_all_layers_activations(hf_model, hf_input_ids)
    end_time = time.time()

    # Compare activations
    print(f"\nMy model activations shape: {len(my_activations_list)}")
    print(f"\nHugging Face model activations shape: {len(hf_activations_list)}")

    mse_list = np.zeros(len(my_activations_list))
    mae_list = np.zeros(len(my_activations_list))
    cos_sims = np.zeros(len(my_activations_list))

    for i, (my_activations,hf_activations) in enumerate(zip(my_activations_list, hf_activations_list)):
        if my_activations.shape == hf_activations.shape:
            mse = torch.mean((my_activations - hf_activations) ** 2)
            mse_list[i] = mse
            
            mae = torch.mean(torch.abs(my_activations - hf_activations))
            mae_list[i] = mae
            
            cosine_similarity = torch.nn.functional.cosine_similarity(my_activations.flatten(), hf_activations.flatten(), dim=0)
            cos_sims[i] = cosine_similarity
        else:
            print(f"Activation shapes do not match at layer {i}. Cannot compute similarity.")
    
    print("MSE, MAE, COSINE")
    np.set_printoptions(precision=5, suppress=True)
    print(np.vstack((mse_list,mae_list,cos_sims)).T)

    print(f"It took {end_time-start_time:.2f} seconds to compare activations")
    
    # Get outputs
    print("Trying to generate")
    start_time = time.time()
    with torch.no_grad():
        prompt = [prompt]
        my_outputs, hf_outputs = generate(my_llama,my_tokenizer, hf_model, hf_tokenizer, prompt)
        print(f"My output:\n{prompt[0] + my_outputs[0]['generation']}")
        print(f"HF output:\n{hf_outputs}")
    end_time = time.time()
    print(f"It took {end_time-start_time:.2f} seconds to generate and compare outputs")

def compare_last_layer_activations(input_prompt):
    # Initialize my implementation
    ckpt_dir = "/home/huang717/.llama/checkpoints/Llama-3-8B/"
    tokenizer_path = "/home/huang717/.llama/checkpoints/Llama-3-8B/tokenizer.model"
    max_seq_len = 128
    max_batch_size = 1
    my_llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=1
    )
    my_model = my_llama.model.cuda()
    my_tokenizer = my_llama.tokenizer
    my_model.return_activation = True  # Enable returning activation

    # Initialize Hugging Face implementation
    print("Trying to load Hugging Face model")
    hf_path = "/home/huang717/.llama/checkpoints/Llama-3-8B-HF"
    # hf_model = LlamaForCausalLM.from_pretrained(
    #     hf_path,
    #     torch_dtype=torch.bfloat16).cuda()
    # hf_config = AutoConfig.from_pretrained(hf_path)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_path).cuda()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_path)

    # Set both models to evaluation mode
    my_model.eval()
    hf_model.eval()

    # Prepare input
    prompt = input_prompt
    my_input_ids = torch.tensor([my_tokenizer.encode(prompt, bos=True, eos=False)]).cuda()
    hf_input_ids = hf_tokenizer.encode(prompt, return_tensors="pt").cuda()
    print(f"Check the input tokens the same -- {my_input_ids==hf_input_ids}")

    # Get activations
    print("Trying to get activations")

    start_time = time.time()
    with torch.no_grad():
        my_activations = get_all_layers_activations(my_model, my_input_ids)[-1]
        hf_activations = get_all_layers_activations(hf_model, hf_input_ids)[-1]
    end_time = time.time()

    # Compare activations
    print(f"My model activations shape: {my_activations.shape}")
    print(f"Hugging Face model activations shape: {hf_activations.shape}")

    if my_activations.shape == hf_activations.shape:
        mse = torch.mean((my_activations - hf_activations) ** 2)
        print(f"Mean Squared Error between activations: {mse.item()}")
        
        mae = torch.mean(torch.abs(my_activations - hf_activations))
        print(f"Mean Absolute Error between activations: {mae.item()}")
        
        cosine_similarity = torch.nn.functional.cosine_similarity(my_activations.flatten(), hf_activations.flatten(), dim=0)
        print(f"Cosine similarity between activations: {cosine_similarity.item()}")

        print(my_activations)
        print(hf_activations)
    else:
        print("Activation shapes do not match. Cannot compute similarity.")

    print(f"It took {end_time-start_time:.2f} seconds to compare activations")
    
    # Get outputs
    print("Trying to generate")
    start_time = time.time()
    with torch.no_grad():
        prompt = [prompt]
        my_outputs, hf_outputs = generate(my_llama,my_tokenizer, hf_model, hf_tokenizer, prompt)
        print(f"My output:\n{prompt[0] + my_outputs[0]['generation']}")
        print(f"HF output:\n{hf_outputs}")
    end_time = time.time()
    print(f"It took {end_time-start_time:.2f} seconds to generate and compare outputs")

def generate_sentence():
    subjects = ["The cat", "A dog", "My friend", "The elephant", "A bird"]
    verbs = ["runs", "jumps", "sings", "dances", "writes"]
    objects = ["quickly", "happily", "in the park", "with enthusiasm", "on the table"]

    subject = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)

    return f"{subject} {verb} {obj}."

if __name__ == "__main__":
    prompt = "The Theory of Universal Approximation states that"
    # prompt = "The quick brown fox jumps over the lazy dog.",
    # prompt = "The hand sanitizer was actually clear glue.",
    #     "He walked into the basement with the horror movie from the night before playing in his head.",
    #     "It's important to remember to be aware of rampaging grizzly bears.",
    #     "The blue parrot drove by the hitchhiking mongoose."
    # # Generate and print 5 random sentences
    # for _ in range(5):
    #     prompt = generate_sentence()
    #     break


    compare_all_layer_activations(prompt)
    # compare_last_layer_activations(prompt)