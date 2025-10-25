"""
Data loading utilities for Deep Sets training.
"""

from pathlib import Path
from typing import Optional, Dict
import sys
import os

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import after path is set
import dataset.hackathon as hackathon_module
import dataset.collate as collate_module
from deepset.config import DataConfig


def get_dataloaders(
    config: DataConfig,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: DataConfig with data parameters
        batch_size: Batch size
        num_workers: Number of worker processes
    
    Returns:
        dataloaders: Dictionary with 'train', 'val', 'test' loaders
    """
    # Create datasets
    print(f"Loading data from: {config.data_root}")
    
    train_dataset = hackathon_module.HackathonDataset(
        split="train",
        root=config.data_root,
        sampling_strategy=config.sampling_strategy,
        seed=config.seed,
    )
    
    val_dataset = hackathon_module.HackathonDataset(
        split="val",
        root=config.data_root,
        sampling_strategy=config.sampling_strategy,
        seed=config.seed,
    )
    
    test_dataset = hackathon_module.HackathonDataset(
        split="test",
        root=config.data_root,
        sampling_strategy=None,  # No sampling for test
        seed=config.seed,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_module.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_module.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_module.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def get_single_batch(dataloader: DataLoader) -> Dict[str, torch.Tensor]:
    """Get a single batch from dataloader for testing."""
    return next(iter(dataloader))


if __name__ == "__main__":
    # Test data loading
    from deepset.config import get_config
    
    print("Testing data loading...")
    
    config = get_config("debug")
    dataloaders = get_dataloaders(
        config.data,
        batch_size=config.training.batch_size,
        num_workers=0,  # Single process for testing
    )
    
    # Test train loader
    print("\nTesting train loader...")
    batch = get_single_batch(dataloaders["train"])
    
    print(f"Batch keys: {batch.keys()}")
    print(f"X shape: {batch['X'].shape}")
    print(f"Y shape: {batch['Y'].shape}")
    print(f"context shape: {batch['context'].shape}")
    print(f"context_mask shape: {batch['context_mask'].shape}")
    
    # Check data statistics
    print(f"\nX range: [{batch['X'].min():.2f}, {batch['X'].max():.2f}]")
    print(f"Y positive rate: {batch['Y'].float().mean():.4f}")
    print(f"Context mask fill rate: {batch['context_mask'].float().mean():.4f}")
    
    # Test a few batches
    print("\nIterating through 3 batches...")
    for i, batch in enumerate(dataloaders["train"]):
        if i >= 3:
            break
        print(f"Batch {i}: X={batch['X'].shape}, Y={batch['Y'].shape}")
    
    print("\n✓ Data loading test passed!")
