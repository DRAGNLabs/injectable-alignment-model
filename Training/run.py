from main import LLaMA
#from config import train_config
import config

def main():
    # TODO: Take in cmdline arg for config name?

    # Declare desired config
    train_args = config.train_config() # You can customize this here

    # Build model class
    Drew_and_Jay_and_Jacksons_Llama = LLaMA.build(train_args=train_args)
    
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