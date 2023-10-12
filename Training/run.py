from main import LLaMA

def main():
    # TODO: Take in cmdline arg for config name

    # TODO: fix all file paths
    # TODO: this doesn't make sense..
    path_to_dataset = "../../Dataset/tokenized/toy_tokenized_data.pkl"
    ckpt_dir = ""
    #TODO: this kinda breaks/doesn't make sense because this path is ultimately used inside a different directory, need to find another way.
    tokenizer_path = "../../Tokenizers/tokenizer.model"
    max_seq_len = 1024
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