import os
from typing import Dict, Any, Optional
import json
from datetime import datetime
import csv

from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.logger import rank_zero_experiment

class CustomTrainingLogger(Logger):
    def __init__(self, save_dir: str, name: str, version: Optional[str] = None):
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self._version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create logging directory
        self.log_dir = os.path.join(save_dir, name, f"version_{self._version}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize files for different metrics
        self.train_loss_file = os.path.join(self.log_dir, "train_loss_per_step.csv")
        self.val_loss_file = os.path.join(self.log_dir, "validation_metrics.csv")
        self.epoch_metrics_file = os.path.join(self.log_dir, "epoch_metrics.csv")
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
        # Keep track of current epoch
        self.current_epoch = 0

    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Training loss file
        with open(self.train_loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'loss', 'learning_rate', 'timestamp'])
            
        # Validation metrics file
        with open(self.val_loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'val_loss', 'timestamp'])
            
        # Epoch metrics file
        with open(self.epoch_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'avg_train_loss', 'avg_val_loss', 'timestamp'])

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @rank_zero_experiment
    def experiment(self):
        return None

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to a JSON file"""
        params_file = os.path.join(self.log_dir, "hyperparameters.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=4)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics based on their names"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log training loss per step
        if 'train_loss' in metrics:
            with open(self.train_loss_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_epoch,
                    step,
                    metrics['train_loss'],
                    metrics.get('learning_rate', ''),
                    timestamp
                ])
        
        # Log validation metrics
        if 'val_loss' in metrics:
            with open(self.val_loss_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_epoch,
                    step,
                    metrics['val_loss'],
                    timestamp
                ])
        
        # Log epoch metrics
        if 'epoch' in metrics:
            self.current_epoch = metrics['epoch']
            with open(self.epoch_metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_epoch,
                    metrics.get('avg_train_loss', ''),
                    metrics.get('avg_val_loss', ''),
                    timestamp
                ])

    def save(self) -> None:
        """Save any custom state if needed"""
        pass

# Example usage in the training script:
"""
# Initialize the logger
custom_logger = CustomTrainingLogger(
    save_dir='logs',
    name='custom_experiment'
)

# Add to your trainer configuration
trainer = Trainer(
    logger=custom_logger,
    # ... other trainer args
)

# In your LightningModule, you can log metrics:
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log('train_loss', loss, on_step=True, on_epoch=True)
    return loss

def validation_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log('val_loss', loss, on_step=True, on_epoch=True)
    return loss
"""