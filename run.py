from llama import LLaMA
import sys
from utils.data_utils import Struct
import yaml

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    args = Struct(**args)

    # Build model class
    Drew_and_Jay_and_Jacksons_Llama = LLaMA.build(train_args=args)
    
    # Train
    Drew_and_Jay_and_Jacksons_Llama.train()

    # Generate
    prompt = ["test test test"]
    max_gen_len = 10
    
    decoded, dictionary = Drew_and_Jay_and_Jacksons_Llama.generate(prompt, max_gen_len, repetition_penalty=9.0)

    print('decoded: ', decoded)
    print('dictionary: ', dictionary)

    print('\nNo errors!\n')

if __name__ == "__main__":
    main()