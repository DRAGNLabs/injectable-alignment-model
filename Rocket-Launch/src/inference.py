import signal
import sys
import yaml
import torch

from lightning.dataset import DataModule
from lightning.model import Model
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from sp_tokenizer.tokenizer import Tokenizer as SPTokenizer
from transformers import PreTrainedTokenizerFast as HFTokenizer
from utils.data_utils import Struct

def inference(config):
    print('Beginning Inference')
    
    if config.tokenizer_type == 'hf':
        tokenizer = HFTokenizer.from_pretrained(config.tokenizer_path, padding_size='left')
        config.pad_id = tokenizer.pad_token_id
    elif config.tokenizer_type == 'sp':
        tokenizer = SPTokenizer(config.tokenizer_path)
        config.vocab_size = tokenizer.n_words
        config.pad_id = tokenizer.pad_id
    else:
        raise ValueError(f"Tokenizer type '{config.tokenizer_type}' not recognized. Must be 'hf' or 'sp'.")

    # Build model class
    model = Model(tokenizer=tokenizer, config=config)

    csv_logger = CSVLogger(save_dir=config.default_root_dir, name='csv_logs')

    tb_logger = TensorBoardLogger(save_dir=config.default_root_dir, name='tb_logs')

    # Load checkpoint
    checkpoint_path=config.checkpoint_path

    print(f"Using checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])

    dm = DataModule(config, tokenizer)

    # Train
    if not config.use_slurm:
        trainer = Trainer(
            default_root_dir=config.default_root_dir,
            accelerator=config.accelerator,
            val_check_interval=config.val_check_interval,
            log_every_n_steps=config.log_every_n_steps,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            max_epochs=config.num_epochs,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            sync_batchnorm=True,
            logger=[csv_logger, tb_logger]
            )
    else:
        trainer = Trainer(
            default_root_dir=config.default_root_dir,
            accelerator=config.accelerator,
            val_check_interval=config.val_check_interval,
            log_every_n_steps=config.log_every_n_steps,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            num_nodes=config.num_nodes,
            devices=config.devices,
            strategy="ddp",
            max_epochs=config.num_epochs,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            sync_batchnorm=True,
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
            logger=[csv_logger, tb_logger]
            )
    
    print("\nTesting model...")
    trainer.test(model, datamodule=dm)

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert args dict to object
    config = Struct(**config)

    inference(config)

if __name__ == "__main__":
    main()
    