import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pathlib import Path

from tokenizer.tokenizer import Tokenizer
from injected_model import Transformer

# from model import Transformer


# Use a lower precision for better performance
torch.set_float32_matmul_precision('medium')

class LLaMAI(LightningModule):
    def __init__(self,
                 tokenizer: Tokenizer, 
                 config: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.IRM_layers = config.IRM_layers
        self.model = Transformer(config)
        self.validation_step_outputs = [] # Used for saving predictions throughout training

    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        (x, y_true) = batch
        #with autocast(): # autocast is torch package for running in mixed precision, which improves performance
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y_true)

        loss = loss/self.config.gradient_accumulation_steps

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y_true) = batch
        y_hat = self.model(x)
        eval_loss = F.cross_entropy(y_hat, y_true)

        if self.config.save_predictions_during_training:
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(y_hat, 1).detach().cpu().tolist()

            decoded = self.tokenizer.decode(preds)

            self.validation_step_outputs.append(decoded)

        perplexity = torch.exp(eval_loss)
        self.log('val_loss', eval_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_perplexity', perplexity, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return eval_loss

    def on_validation_epoch_end(self) -> None:
        if self.config.save_predictions_during_training == True:
            dir_path = Path(self.config.default_root_dir)
            file_path = dir_path / 'validation_predictions.txt'

            # Check if the directory exists. If not, create it
            dir_path.mkdir(parents=True, exist_ok=True)

            # Check if the file exists. If not, create it and append the outputs
            with file_path.open('a') as f:
                f.write(str(self.validation_step_outputs) + '\n')

            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        params = []
        for layer in self.IRM_layers: params += list(self.model.layers[layer].IRM.parameters())
        optimizer = torch.optim.Adam(params, lr=self.config.lr)  # model.paramaters = weights tensor

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer], [lr_scheduler]
