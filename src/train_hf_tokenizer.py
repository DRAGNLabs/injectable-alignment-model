from pathlib import Path
import sys
import yaml

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, pre_tokenizers, processors, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from utils.data_utils import Struct

def train_hf_tokenizer(config: Struct):
    """
    Modify this function to train a tokenizer using the HuggingFace tokenizers library.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    if config.vocab_size <= 0:
        raise ValueError("Configuration parameter 'vocab_size' must be defined in order to train HF tokenizer.")

    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        show_progress=True,
        special_tokens=["<pad>", "<bos>", "<unk>", "<eos>"])
    
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    print('Beginning training of tokenizer...')
    if config.raw_train_path is not None:
        tokenizer.train([config.raw_train_path], trainer)
    else:
        raise ValueError("Configuration parameter 'raw_train_path' must be defined in order to train HF tokenizer.")

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    tokenizer.decoder = decoders.ByteLevel()

    # # Enable padding
    # tokenizer.enable_padding(
    #     direction="right",
    #     pad_id=0,
    #     pad_token="<pad>",
    #     length=config.max_sequence_embeddings + 1)

    # Enable truncation
    tokenizer.enable_truncation(
        max_length=config.max_sequence_embeddings + 1,
        direction="right")

    # Wrap tokenizer with transformers library
    tokenizer = PreTrainedTokenizerFast(
        model_max_length=config.max_sequence_embeddings,
        padding_side="right",
        truncation_side="right",
        bos_token="<bos>",
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        tokenizer_object=tokenizer)

    # Save tokenizer to file
    tokenizer_save_path = Path(config.tokenizer_path)
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_path)
    print('Finished!')

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert args dict to object
    config = Struct(**config)

    train_hf_tokenizer(config)

if __name__ == '__main__':
    main()
