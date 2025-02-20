import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, PreTrainedTokenizer

@dataclass
class TrainerConfig:
    num_epochs: int
    gradient_accumulation_steps: int
    val_check_interval: float
    save_top_k: int
    checkpoint_dir: str
    log_dir: str
    precision: str = "bf16"
    
class CustomTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        self.writer = SummaryWriter(config.log_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = GradScaler() if config.precision == "fp16" else None
        self.best_val_loss = float('inf')
        self.best_checkpoints = []
        
        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[device])
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        checkpoint_path = self.checkpoint_dir / f"model-epoch{epoch}-loss{val_loss:.3f}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.best_checkpoints.append((val_loss, checkpoint_path))
        self.best_checkpoints.sort(key=lambda x: x[0])
        
        # Keep only top k checkpoints
        while len(self.best_checkpoints) > self.config.save_top_k:
            _, checkpoint_to_remove = self.best_checkpoints.pop()
            if checkpoint_to_remove.exists():
                checkpoint_to_remove.unlink()
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with autocast(device_type=self.device.type, 
                         dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float16):
                loss = self.model(**batch).loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights if gradient accumulation steps reached
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Log training progress
            if batch_idx % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), 
                                     epoch * num_batches + batch_idx)
            
            # Run validation if needed
            steps_done = epoch * num_batches + batch_idx
            if self.config.val_check_interval > 0 and \
               steps_done % int(num_batches * self.config.val_check_interval) == 0:
                val_loss = self.validate()
                self.model.train()
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss)
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with autocast(device_type=self.device.type,
                         dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float16):
                loss = self.model(**batch).loss
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('val/loss', avg_loss, self.model.global_step)
        return avg_loss
    
    def train(self):
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            epoch_time = time.time() - start_time
            
            # Log epoch metrics
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time
            }
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, time={epoch_time:.2f}s")
            
            with open(self.config.log_dir / 'metrics.jsonl', 'a') as f:
                f.write(json.dumps(metrics) + '\n')

def train_model(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    config: TrainerConfig
):
    # Setup device and distributed training
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if os.environ.get('SLURM_PROCID'):
            dist.init_process_group('nccl')
            local_rank = int(os.environ['LOCAL_RANK'])
            device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=not dist.is_initialized(),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate
    )
    
    # Create and run trainer
    trainer = CustomTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
        device=device
    )
    
    trainer.train()

if __name__ == "__main__":
    # Example usage
    config = TrainerConfig(
        num_epochs=10,
        gradient_accumulation_steps=4,
        val_check_interval=0.5,
        save_top_k=3,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        precision="bf16"
    )
    
    # Initialize model, datasets, and start training
    train_model(model, train_dataset, val_dataset, config)