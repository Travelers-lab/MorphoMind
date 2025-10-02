#!/usr/bin/env python3
"""
Main training script for MorphoModel.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoint.pth
    python train.py --help
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.backends.cudnn as cudnn

from .config import TrainingConfig
from dataloader import create_dataloaders, CSVDataloader
from model import create_model
from loss import create_loss_function
from optimizer import create_optimizer_and_scheduler
from trainer import Trainer
from utils import (
    set_seed, setup_logging, save_checkpoint, load_checkpoint,
    plot_training_history, save_experiment_config, print_model_summary
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MorphoModel')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (overrides config)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run single batch for debugging')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Load configuration
    if os.path.exists(args.config):
        config = TrainingConfig.from_yaml(args.config)
    else:
        print(f"Config file {args.config} not found. Using default configuration.")
        config = TrainingConfig()
    
    # Override config with command line arguments
    if args.device is not None:
        config.device = args.device
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = 'cpu'
    
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    if config.device == 'cuda':
        cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Setup logging
    logger = setup_logging(config.log_dir, config.experiment_name)
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    # Save experiment configuration
    config_save_path = os.path.join(config.log_dir, f'{config.experiment_name}_config.json')
    save_experiment_config(config, config_save_path)
    
    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_dataloaders(config)
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
        if test_loader:
            logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        model = model.to(device)
        
        # Print model summary
        print_model_summary(model, config)
        
        # Create loss function
        loss_function = create_loss_function(config)
        loss_function = loss_function.to(device)
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
            start_epoch = checkpoint.get('epoch', 0) + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            config=config,
            device=device
        )
        
        # Set starting epoch
        trainer.current_epoch = start_epoch
        
        # Dry run for debugging
        if args.dry_run:
            logger.info("Running dry run (single batch)...")
            config.num_epochs = start_epoch + 1
            config.log_frequency = 1
        
        # Start training
        logger.info("Starting training...")
        history = trainer.train()
        
        # Save final model
        final_checkpoint_path = os.path.join(config.checkpoint_dir, 'final_model.pth')
        save_checkpoint({
            'epoch': trainer.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            'train_history': history['train_history'],
            'val_history': history['val_history'],
            'config': config
        }, final_checkpoint_path)
        
        # Plot training history
        plot_path = os.path.join(config.log_dir, f'{config.experiment_name}_training_history.png')
        plot_training_history(history['train_history'], history['val_history'], plot_path)
        
        # Test evaluation if test set is available
        if test_loader:
            logger.info("Evaluating on test set...")
            trainer.val_loader = test_loader
            test_metrics = trainer.validate()
            logger.info(f"Test Results: {test_metrics}")
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save interrupted checkpoint
        interrupted_checkpoint_path = os.path.join(config.checkpoint_dir, 'interrupted_checkpoint.pth')
        save_checkpoint({
            'epoch': trainer.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            'config': config
        }, interrupted_checkpoint_path)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()