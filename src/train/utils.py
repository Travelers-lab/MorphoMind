import torch
import numpy as np
import random
import os
import logging
import json
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'{experiment_name}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('MorphoModel')
    return logger


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def save_checkpoint(state: Dict[str, Any], filename: str):
    """Save model checkpoint."""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename: str, model, optimizer=None, scheduler=None) -> Dict[str, Any]:
    """Load model checkpoint."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")
    
    checkpoint = torch.load(filename, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded: {filename}")
    return checkpoint


def plot_training_history(train_history: Dict, val_history: Dict, save_path: str):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    axes[0, 0].plot(train_history['loss'], label='Train')
    axes[0, 0].plot(val_history['loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Learning rate plot
    axes[0, 1].plot(train_history['lr'])
    axes[0, 1].set_title('Learning Rate')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('LR')
    axes[0, 1].grid(True)
    
    # Component losses if available
    if 'position_loss' in val_history:
        axes[1, 0].plot(val_history['position_loss'], label='Position')
        axes[1, 0].plot(val_history['velocity_loss'], label='Velocity')
        axes[1, 0].plot(val_history['torque_loss'], label='Torque')
        axes[1, 0].set_title('Component Losses')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_model_flops(model, input_shape, device='cpu'):
    """Calculate FLOPs for the model (requires thop library)."""
    try:
        from thop import profile
        
        # Create dummy inputs
        X = torch.randn(1, *input_shape[0])  # X input
        command = torch.randn(1, *input_shape[1])  # command input
        
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
            X, command = X.cuda(), command.cuda()
        
        flops, params = profile(model, inputs=(X, command), verbose=False)
        return flops, params
    
    except ImportError:
        print("Warning: thop library not installed. Cannot calculate FLOPs.")
        return None, None


def save_experiment_config(config, save_path: str):
    """Save experiment configuration to JSON."""
    config_dict = {}
    for key, value in config.__dict__.items():
        if isinstance(value, (int, float, str, bool, list, dict)) or value is None:
            config_dict[key] = value
        else:
            config_dict[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def print_model_summary(model, config):
    """Print model summary information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Input shapes
    print(f"Input X shape: {config.sequence_length} x {config.input_features}")
    print(f"Input command shape: {config.sequence_length} x {config.command_features}")
    print(f"Output shape: 3 x {config.num_joints}")
    print("=" * 50)