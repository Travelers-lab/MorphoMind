import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MorphoLoss(nn.Module):
    """
    Custom loss function for MorphoModel.
    Combines position, velocity, and torque losses with configurable weights.
    """
    
    def __init__(
        self,
        position_weight: float = 1.0,
        velocity_weight: float = 1.0,
        torque_weight: float = 1.0,
        loss_type: str = 'mse'
    ):
        """
        Initialize MorphoLoss.
        
        Args:
            position_weight: Weight for position loss
            velocity_weight: Weight for velocity loss  
            torque_weight: Weight for torque loss
            loss_type: Type of base loss ('mse', 'huber', 'smooth_l1')
        """
        super(MorphoLoss, self).__init__()
        
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.torque_weight = torque_weight
        
        # Select base loss function
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='mean')
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss(reduction='mean', delta=1.0)
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss between predictions and targets.
        
        Args:
            predictions: Model output [batch_size, 3, 12]
            targets: Ground truth [batch_size, 3, 12]
            
        Returns:
            total_loss: Combined weighted loss
            loss_components: Dictionary of individual loss components
        """
        # Split into position, velocity, torque components
        pred_position = predictions[:, 0, :]  # [batch_size, 12]
        pred_velocity = predictions[:, 1, :]  # [batch_size, 12]
        pred_torque = predictions[:, 2, :]    # [batch_size, 12]
        
        target_position = targets[:, 0, :]    # [batch_size, 12]
        target_velocity = targets[:, 1, :]    # [batch_size, 12]
        target_torque = targets[:, 2, :]      # [batch_size, 12]
        
        # Compute individual losses
        position_loss = self.base_loss(pred_position, target_position)
        velocity_loss = self.base_loss(pred_velocity, target_velocity)
        torque_loss = self.base_loss(pred_torque, target_torque)
        
        # Weighted combination
        total_loss = (
            self.position_weight * position_loss +
            self.velocity_weight * velocity_loss +
            self.torque_weight * torque_loss
        )
        
        # Loss components for logging
        loss_components = {
            'position_loss': position_loss.item(),
            'velocity_loss': velocity_loss.item(),
            'torque_loss': torque_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss with learnable loss weights.
    Automatically balances different loss components during training.
    """
    
    def __init__(self, num_losses: int = 3):
        """
        Initialize adaptive loss.
        
        Args:
            num_losses: Number of loss components (position, velocity, torque)
        """
        super(AdaptiveLoss, self).__init__()
        
        # Learnable loss weights (log space for numerical stability)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        self.base_loss = nn.MSELoss(reduction='mean')
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute adaptive weighted loss.
        
        Args:
            predictions: Model output [batch_size, 3, 12]
            targets: Ground truth [batch_size, 3, 12]
            
        Returns:
            total_loss: Adaptively weighted loss
            loss_components: Dictionary of loss components and weights
        """
        # Compute individual losses
        losses = []
        for i in range(3):  # position, velocity, torque
            loss = self.base_loss(predictions[:, i, :], targets[:, i, :])
            losses.append(loss)
        
        losses = torch.stack(losses)
        
        # Compute adaptive weights
        precision = torch.exp(-self.log_vars)
        
        # Adaptive loss: precision * loss + log_var (uncertainty regularization)
        adaptive_losses = precision * losses + self.log_vars
        total_loss = adaptive_losses.sum()
        
        # Convert to weights for interpretability
        weights = precision / precision.sum()
        
        loss_components = {
            'position_loss': losses[0].item(),
            'velocity_loss': losses[1].item(), 
            'torque_loss': losses[2].item(),
            'total_loss': total_loss.item(),
            'position_weight': weights[0].item(),
            'velocity_weight': weights[1].item(),
            'torque_weight': weights[2].item()
        }
        
        return total_loss, loss_components


def create_loss_function(config) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Loss function instance
    """
    if hasattr(config, 'adaptive_loss') and config.adaptive_loss:
        return AdaptiveLoss(num_losses=3)
    else:
        return MorphoLoss(
            position_weight=config.position_loss_weight,
            velocity_weight=config.velocity_loss_weight,
            torque_weight=config.torque_loss_weight
        )