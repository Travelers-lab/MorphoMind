import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import yaml


@dataclass
class TrainingConfig:
    """Configuration class for MorphoModel training."""
    
    # Model parameters
    hidden_dim: int = 64
    fusion_layers: int = 4
    backbone_layers: int = 4
    decoder_backbone_layers: int = 4
    dropout: float = 0.1
    
    # Training hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    
    # Data parameters
    sequence_length: int = 128  # n dimension
    input_features: int = 25    # X input features
    command_features: int = 9   # command input features
    num_joints: int = 12
    
    # Optimizer settings
    optimizer_type: str = 'adamw'
    scheduler_type: str = 'cosine'
    scheduler_params: Dict[str, Any] = None
    
    # Loss function weights
    position_loss_weight: float = 1.0
    velocity_loss_weight: float = 1.0
    torque_loss_weight: float = 1.0
    
    # Data paths
    data_root: str = './data'
    train_data_path: str = './data/train'
    val_data_path: str = './data/val'
    test_data_path: str = './data/test'
    
    # Experiment settings
    experiment_name: str = 'morpho_model_exp'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    save_frequency: int = 10
    
    # Training settings
    device: str = 'cuda'
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    
    # Validation and early stopping
    val_frequency: int = 1
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Logging
    log_frequency: int = 100
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    wandb_project: str = 'morpho_model'
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.scheduler_params is None:
            self.scheduler_params = {'T_max': self.num_epochs}
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)