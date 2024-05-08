from utils.data_utils import Struct
import sys
import yaml
from tokenizer.tokenizer import Tokenizer
from dataset import DataSet
from torch.utils.data import DataLoader

def main():
    """
    Loads a dataset and prints the memory usage of the dataframe.
    This is outdated, but useful for understanding the structure of the dataset.
    """
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    tokenizer = Tokenizer(model_path=config.tokenizer_path)
    config.vocab_size = tokenizer.n_words
    config.pad_id = tokenizer.pad_id

    train_dataset = DataSet(config.train_path, 
                            pad_tok=tokenizer.pad_id, 
                            bos_tok=tokenizer.bos_id, 
                            eos_tok=tokenizer.eos_id, 
                            max_sequence_embeddings=config.max_sequence_embeddings)
    
    DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # print memory usage of dataframe in train_dataset
    print(train_dataset.data.memory_usage(index=True).sum())

    for i, batch in enumerate(train_dataset):
        if i > 200:
            break
        x, y = batch
        print(x.shape)

if __name__ == "__main__":
    main()
    