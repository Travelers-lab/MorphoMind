import torch
import torch.optim as optim
from torch.optim.lr_scheduler import *
from typing import Dict, Any


def create_optimizer(model, config) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Training configuration
        
    Returns:
        Configured optimizer
    """
    optimizer_type = config.optimizer_type.lower()
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
            nesterov=True
        )
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, config) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config.scheduler_type.lower()
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_params.get('T_max', config.num_epochs),
            eta_min=config.scheduler_params.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'cosine_warm':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler_params.get('T_0', 10),
            T_mult=config.scheduler_params.get('T_mult', 2)
        )
    elif scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config.scheduler_params.get('step_size', 30),
            gamma=config.scheduler_params.get('gamma', 0.1)
        )
    elif scheduler_type == 'multistep':
        scheduler = MultiStepLR(
            optimizer,
            milestones=config.scheduler_params.get('milestones', [30, 60, 90]),
            gamma=config.scheduler_params.get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_params.get('factor', 0.5),
            patience=config.scheduler_params.get('patience', 10),
            verbose=True
        )
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(
            optimizer,
            gamma=config.scheduler_params.get('gamma', 0.95)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


class WarmupScheduler:
    """
    Learning rate warmup scheduler.
    Gradually increases learning rate from 0 to target LR over warmup epochs.
    """
    
    def __init__(self, optimizer, warmup_epochs: int, base_scheduler=None):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            base_scheduler: Main scheduler to use after warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * lr_scale
        elif self.base_scheduler is not None:
            # Main scheduling phase
            self.base_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


def create_optimizer_and_scheduler(model, config):
    """
    Create optimizer and scheduler with optional warmup.
    
    Args:
        model: PyTorch model
        config: Training configuration
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = create_optimizer(model, config)
    base_scheduler = create_scheduler(optimizer, config)
    
    # Add warmup if specified
    if config.warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer, 
            warmup_epochs=config.warmup_epochs,
            base_scheduler=base_scheduler
        )
    else:
        scheduler = base_scheduler
    
    return optimizer, scheduler