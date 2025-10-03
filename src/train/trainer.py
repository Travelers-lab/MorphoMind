import torch
import pandas as pd
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import time
from typing import Dict, Tuple, Optional
from .utils import AverageMeter, save_checkpoint
from dataloader import CSVDataloader

class Trainer:
    """Training manager for MorphoModel."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_function,
        config,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            model: MorphoModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            loss_function: Loss function
            config: Training configuration
            device: Training device
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.config = config
        self.device = device
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Metrics tracking
        self.train_history = {'loss': [], 'lr': []}
        self.val_history = {'loss': []}
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Metrics
        loss_meter = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        end_time = time.time()
        for chunk_id,  chunk in enumerate(pd.read_csv(self.config.train_data_path, chunksize=self.config.chunk_size)):
            chunk = pd.DataFrame(chunk.values, columns=chunk.columns)
            datasets = CSVDataloader(chunk)
            data_loader = DataLoader(datasets, batch_size=self.config.batch_size, shuffle=True,
                                      num_workers=4)
            for batch_idx, batch in enumerate(data_loader):
                # Data loading time
                data_time.update(time.time() - end_time)

                # Move data to device
                X = batch['X'].to(self.device, non_blocking=True)
                command = batch['command'].to(self.device, non_blocking=True)
                target = batch['target'].to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                if self.config.mixed_precision:
                    with autocast():
                        predictions = self.model(X, command)
                        loss, loss_components = self.loss_function(predictions, target)
                else:
                    predictions = self.model(X, command)
                    loss, loss_components = self.loss_function(predictions, target)

                # Backward pass
                self.optimizer.zero_grad()

                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()

                    # Gradient clipping
                    if self.config.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.grad_clip_norm
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()

                    # Gradient clipping
                    if self.config.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.grad_clip_norm
                        )

                    self.optimizer.step()

                # Update metrics
                loss_meter.update(loss.item(), X.size(0))
                batch_time.update(time.time() - end_time)
                end_time = time.time()

                # Logging
                if batch_idx % self.config.log_frequency == 0:
                    print(f'Train Epoch: {self.current_epoch} '
                          f'[{batch_idx}/{len(self.train_loader)} '
                          f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                          f'Loss: {loss.item():.6f} '
                          f'Data: {data_time.avg:.3f}s '
                          f'Batch: {batch_time.avg:.3f}s')
        
        return {
            'loss': loss_meter.avg,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        loss_meter = AverageMeter()
        position_loss_meter = AverageMeter()
        velocity_loss_meter = AverageMeter()
        torque_loss_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                X = batch['X'].to(self.device, non_blocking=True)
                command = batch['command'].to(self.device, non_blocking=True)
                target = batch['target'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.config.mixed_precision:
                    with autocast():
                        predictions = self.model(X, command)
                        loss, loss_components = self.loss_function(predictions, target)
                else:
                    predictions = self.model(X, command)
                    loss, loss_components = self.loss_function(predictions, target)
                
                # Update metrics
                batch_size = X.size(0)
                loss_meter.update(loss.item(), batch_size)
                position_loss_meter.update(loss_components['position_loss'], batch_size)
                velocity_loss_meter.update(loss_components['velocity_loss'], batch_size)
                torque_loss_meter.update(loss_components['torque_loss'], batch_size)
        
        val_metrics = {
            'loss': loss_meter.avg,
            'position_loss': position_loss_meter.avg,
            'velocity_loss': velocity_loss_meter.avg,
            'torque_loss': torque_loss_meter.avg
        }
        
        print(f'Validation - Loss: {loss_meter.avg:.6f}, '
              f'Pos: {position_loss_meter.avg:.6f}, '
              f'Vel: {velocity_loss_meter.avg:.6f}, '
              f'Torque: {torque_loss_meter.avg:.6f}')
        
        return val_metrics
    
    def train(self) -> Dict[str, list]:
        """Full training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Training on device: {self.device}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['lr'].append(train_metrics['lr'])
            
            # Learning rate scheduling
            if hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Plateau scheduler needs validation loss
                    val_metrics = self.validate()
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Validation
            if epoch % self.config.val_frequency == 0:
                val_metrics = self.validate()
                self.val_history['loss'].append(val_metrics['loss'])
                
                # Early stopping check
                if val_metrics['loss'] < self.best_val_loss - self.config.early_stopping_min_delta:
                    self.best_val_loss = val_metrics['loss']
                    self.early_stopping_counter = 0
                    
                    # Save best model
                    save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
                        'best_val_loss': self.best_val_loss,
                        'config': self.config
                    }, 
                    filename=f"{self.config.checkpoint_dir}/best_model.pth"
                    )
                else:
                    self.early_stopping_counter += 1
                
                # Early stopping
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Periodic checkpoint saving
            if epoch % self.config.save_frequency == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
                    'train_history': self.train_history,
                    'val_history': self.val_history,
                    'config': self.config
                }, 
                filename=f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
                )
        
        print("Training completed!")
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }