import sys
import signal
import yaml

import torch
import pickle
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger

from dataset import DataModule
from tokenizer.tokenizer import Tokenizer
from injected_llama import LLaMAI as LLaMA
from utils.data_utils import Struct

torch.set_float32_matmul_precision('medium')

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started")
    def on_train_end(self, trainer, pl_module):
        print("Training ended")

def train(config):
    seed_everything(config.seed, workers=True)
    tokenizer = Tokenizer(model_path=config.tokenizer_path)
    config.vocab_size = tokenizer.n_words
    config.pad_id = tokenizer.pad_id

    # Build model class
    model = LLaMA(tokenizer=tokenizer, config=config)
    with open(config.train_path, "rb") as f:
        stuff=  pickle.load(f)
	
    print("PRINTING PICKLE FILE!!!!!")
    print(stuff)
    
    dm = DataModule(config.train_path, config.eval_path, tokenizer, config.batch_size, config.sequence_length)
    print("\n\nPRINTING DATAMODULE!!!!!")
    print(dm)

    # callbacks
    early_stopping = EarlyStopping('val_loss', patience=config.early_stopping, mode='max', verbose=True)
    logger = CSVLogger(save_dir=config.default_root_dir, name='logs')
    model_checkpoint = ModelCheckpoint(
        dirpath=config.default_root_dir + '/checkpoints',
        filename='model-{epoch}-{val_loss:.2f}',
        save_top_k=config.save_top_k,
        monitor='val_loss',
        mode='max')
    print_callback = PrintCallback()

    # Train
    trainer = Trainer(
        default_root_dir=config.default_root_dir,
        accelerator=config.accelerator,
        num_nodes=config.num_nodes,
        devices=config.devices,
        strategy="ddp",
        max_epochs=config.num_epochs,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        sync_batchnorm=True,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
        callbacks=[early_stopping, print_callback, model_checkpoint],
        logger=logger
        )
    trainer.fit(model, datamodule=dm)

    print('\nNo errors!\n')

def main():
    print(f"GPU COUNT: {torch.cuda.device_count()}")
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    train(config)

if __name__ == "__main__":
    main()