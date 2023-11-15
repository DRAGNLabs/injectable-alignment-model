import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from tokenizer.tokenizer import Tokenizer
from model import Transformer

# Use a lower precision for better performance
torch.set_float32_matmul_precision('medium')

class LLaMA(LightningModule):
    def __init__(self,
                 tokenizer: Tokenizer, 
                 config: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.model = Transformer(config)
        self.validation_step_outputs = [] # Used for saving predictions throughout training

    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        (x, y_true) = batch
        #with autocast(): # autocast is torch package for running in mixed precision, which improves performance
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y_true)

        # TODO: is this necessary?
        loss = loss/self.config.gradient_accumulation_steps

        # TODO: log settings?
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y_true) = batch
        y_hat = self.model(x)
        eval_loss = F.cross_entropy(y_hat, y_true)

        # Decode predictions and add to evaluation predictions list
        preds = torch.argmax(y_hat, 1).detach().cpu().tolist()

        decoded = self.tokenizer.decode(preds)

        self.validation_step_outputs.append(decoded)

        # TODO: do you need to get eval loss for entire epoch?
        perplexity = torch.exp(eval_loss)
        self.log('val_loss', eval_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_perplexity', perplexity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return eval_loss

    def on_validation_epoch_end(self) -> None:
        all_preds = torch.stack(self.validation_step_outputs)
        
        # TODO: do something with predictions, like save them to a file

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)  # model.paramaters = weights tensor
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer], [lr_scheduler]
