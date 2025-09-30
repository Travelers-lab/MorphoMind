import torch
import torch.nn as nn
from typing import Dict, Any
import sys
import os

# Add parent directory to path to import MorphoModel
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from morpho_model import MorphModel  # Import the actual model
except ImportError:
    print("Warning: Could not import MorphModel. Using placeholder.")
    MorphModel = None


class MorphoModelWrapper(nn.Module):
    """
    Wrapper class for MorphoModel with additional functionality for training.
    """
    
    def __init__(self, config):
        """
        Initialize MorphoModel wrapper.
        
        Args:
            config: Training configuration object
        """
        super(MorphoModelWrapper, self).__init__()
        
        self.config = config
        
        if MorphModel is None:
            raise ImportError("MorphModel not found. Please ensure it's properly imported.")
        
        # Initialize the core model
        self.model = MorphModel(
            hidden_dim=config.hidden_dim,
            fusion_layers=config.fusion_layers,
            backbone_layers=config.backbone_layers,
            decoder_backbone_layers=config.decoder_backbone_layers
        )
        
        # Model info
        self.input_shape = (config.sequence_length, config.input_features)  # X shape
        self.command_shape = (config.sequence_length, config.command_features)  # command shape
        self.output_shape = (3, config.num_joints)  # output shape
        
    def forward(self, X: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            X: Input tensor [batch_size, sequence_length, 25]
            command: Command tensor [batch_size, sequence_length, 9]
            
        Returns:
            Joint commands [batch_size, 3, 12]
        """
        return self.model(X, command)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': self.input_shape,
            'command_shape': self.command_shape,
            'output_shape': self.output_shape,
            'hidden_dim': self.config.hidden_dim,
            'fusion_layers': self.config.fusion_layers,
            'backbone_layers': self.config.backbone_layers,
            'decoder_backbone_layers': self.config.decoder_backbone_layers
        }
    
    def initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def create_model(config) -> MorphoModelWrapper:
    """
    Create and initialize MorphoModel.
    
    Args:
        config: Training configuration
        
    Returns:
        Initialized MorphoModelWrapper
    """
    model = MorphoModelWrapper(config)
    model.initialize_weights()
    
    # Move to device
    if torch.cuda.is_available() and config.device == 'cuda':
        model = model.cuda()
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)