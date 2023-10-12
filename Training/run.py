from main import LLaMA
from config import train_config

def main():
    # TODO: Take in cmdline arg for config name

    # TODO: fix all file paths
    # TODO: this doesn't make sense..
    path_to_dataset = "../../Dataset/tokenized/toy_tokenized_data.pkl"
    ckpt_dir = ""
    #TODO: this kinda breaks/doesn't make sense because this path is ultimately used inside a different directory, need to find another way.
    tokenizer_path = "../../Tokenizers/tokenizer.model"

    train_args = train_config() # You can customize this here

    Drew_and_Jay_and_Jacksons_Llama = LLaMA.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        dataset_path=path_to_dataset,
        train_args=train_args
        )
    
    Drew_and_Jay_and_Jacksons_Llama.train()

    print('\nNo errors!\n')

if __name__ == "__main__":
    main()