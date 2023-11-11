from llama import LLaMA
import sys
import signal
from utils.data_utils import Struct
import yaml

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from dataset import DataModule
from tokenizer.tokenizer import Tokenizer

class CustomCSVLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def log_metrics(self, metrics, step=None):
        if 'val_loss' in metrics:
            self.experiment.log_metrics(metrics, step=step)
        super().log_metrics(metrics, step=step)

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started")
    def on_train_end(self, trainer, pl_module):
        print("Training ended")

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    tokenizer = Tokenizer(model_path=config.tokenizer_path)  # including this for the special tokens (i.e. pad)
    config.vocab_size = tokenizer.n_words
    config.pad_id = tokenizer.pad_id

    # Build model class
    Drew_and_Jay_and_Jacksons_Llama = LLaMA(tokenizer=tokenizer, config=config)
    
    dm = DataModule(config.train_path, config.eval_path, tokenizer, config.batch_size, config.sequence_length)

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
        max_epochs=500,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
        callbacks=[early_stopping, print_callback, model_checkpoint],
        logger=logger
        )
    trainer.fit(Drew_and_Jay_and_Jacksons_Llama, dm)

    print('\nNo errors!\n')

if __name__ == "__main__":
    main()