import os
import sys
import signal
import yaml

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from transformers import LlamaTokenizer as HFTokenizer

from lightning.dataset import DataModule
from sp_tokenizer.tokenizer import Tokenizer as SPTokenizer
from llama_models.injected_llama_for_causal import LlamaForCausalLM as Model
from utils.data_utils import Struct
from tokenize_data import tokenize_data

torch.set_float32_matmul_precision("medium")

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started")
    def on_train_end(self, trainer, pl_module):
        print("Training ended")

def train(config):
    seed_everything(config.seed, workers=True)

    # Load tokenizer
    if config.tokenizer_type == "hf":
        tokenizer = HFTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_id = tokenizer.pad_token_id
    elif config.tokenizer_type == "sp":
        tokenizer = SPTokenizer(config.tokenizer_path)
        tokenizer.pad_id = tokenizer.eos_id
        config.vocab_size = tokenizer.n_words
        config.pad_id = tokenizer.pad_id
    else:
        raise ValueError(f"Tokenizer type '{config.tokenizer_type}' not recognized. Must be 'hf' or 'sp'.")

    original_checkpoint_path = config.checkpoint_path
    print(f"Instantiating model")
    model = Model(tokenizer, config)
    
    # Load the model from the original checkpoint with strict=False, so it will only fill in the weights that are in both models, without errors
    print(f"Loading from checkpoint")
    checkpoint = torch.load(original_checkpoint_path,  map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    # print(f"Checkpoint conversion complete.")

    model.to("cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu")

    # Set requires_grad to false for everything
    for param in model.parameters():
        param.requires_grad = False
    # Except the IRM parameters
    for irm_param in model.model.irm.parameters():
        irm_param.requires_grad = True

    dm = DataModule(config, tokenizer)

    # Callbacks
    early_stopping = EarlyStopping("val_loss", patience=config.early_stopping, mode="min", verbose=True)
    csv_logger = CSVLogger(save_dir=config.default_root_dir, name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=config.default_root_dir, name="tb_logs")
    model_checkpoint = ModelCheckpoint(
        dirpath=config.default_root_dir + "/checkpoints",
        filename="model-{epoch}-{val_loss:.2f}",
        save_top_k=config.save_top_k,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1
        )
    print_callback = PrintCallback()

    # Train
    if not config.use_slurm:
        trainer = Trainer(
            accelerator=config.accelerator,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            callbacks=[early_stopping, print_callback, model_checkpoint],
            # check_val_every_n_epoch=config.check_val_every_n_epoch,
            default_root_dir=config.default_root_dir,
            log_every_n_steps=config.log_every_n_steps,
            logger=[csv_logger, tb_logger],
            max_epochs=config.num_epochs,
            sync_batchnorm=True,
            val_check_interval=config.val_check_interval
            )
    else:
        trainer = Trainer(
            accelerator=config.accelerator,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            callbacks=[early_stopping, print_callback, model_checkpoint],
            # check_val_every_n_epoch=config.check_val_every_n_epoch,
            default_root_dir=config.default_root_dir,
            devices=config.devices,
            log_every_n_steps=config.log_every_n_steps,
            logger=[csv_logger, tb_logger],
            max_epochs=config.num_epochs,
            num_nodes=config.num_nodes,
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
            strategy="ddp",
            sync_batchnorm=True,
            val_check_interval=config.val_check_interval,
            )
        
    trainer.fit(model, datamodule=dm)

    print("\nNo errors!\n")

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Print the config file for future reference
    s = "" # I append it all to one string so it'll print out together without interruption when running on multiple nodes
    for k in config:
            if k == "model_config":
                s += f"\n{k}:\n"
                for l in config[k]:
                    s += f"  {l}: {config[k][l]}\n"
            else:
                s += f"{k}: {config[k]}\n"

    print(f"{s}")

    # Convert args dict to object
    config = Struct(**config)

    # Comment out this line if you wish to tokenize the data seperately, so as to not repeat processing
    tokenize_data(config)

    train(config)

if __name__ == "__main__":
    main()
