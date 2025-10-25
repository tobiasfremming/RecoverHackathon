"""
Quick test script to validate the entire pipeline.
"""

import sys
from pathlib import Path
import torch

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from deepset.config import get_config
from deepset.model import DeepSetsModel
from deepset.data_loader import get_dataloaders
from deepset.utils import FocalLoss, compute_metrics, set_seed


def test_model():
    """Test model forward pass."""
    print("=" * 80)
    print("Testing Model")
    print("=" * 80)
    
    config = get_config("debug")
    model = DeepSetsModel(
        num_operations=config.model.num_operations,
        num_rooms=config.model.num_rooms,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        pooling=config.model.pooling,
    )
    
    # Test forward pass
    batch_size = 4
    X = torch.randn(batch_size, 399)
    context = torch.randn(batch_size, 10, 399)
    context_mask = torch.ones(batch_size, 10).bool()
    
    logits = model(X, context, context_mask)
    
    print(f"Input shape: {X.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"✓ Model test passed!\n")


def test_data_loading():
    """Test data loading."""
    print("=" * 80)
    print("Testing Data Loading")
    print("=" * 80)
    
    config = get_config("debug")
    
    # Create dataloaders
    dataloaders = get_dataloaders(
        config.data,
        batch_size=config.training.batch_size,
        num_workers=0,  # Single process
    )
    
    # Get a batch
    batch = next(iter(dataloaders["train"]))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"X shape: {batch['X'].shape}")
    print(f"Y shape: {batch['Y'].shape}")
    print(f"context shape: {batch['context'].shape}")
    print(f"context_mask shape: {batch['context_mask'].shape}")
    print(f"Positive rate: {batch['Y'].float().mean():.4f}")
    print(f"✓ Data loading test passed!\n")
    
    return dataloaders


def test_loss():
    """Test loss function."""
    print("=" * 80)
    print("Testing Loss Function")
    print("=" * 80)
    
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
    
    logits = torch.randn(4, 388)
    targets = torch.randint(0, 2, (4, 388))
    
    loss = loss_fn(logits, targets)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"✓ Loss test passed!\n")


def test_training_step():
    """Test a single training step."""
    print("=" * 80)
    print("Testing Training Step")
    print("=" * 80)
    
    set_seed(42)
    
    config = get_config("debug")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = DeepSetsModel(
        num_operations=config.model.num_operations,
        num_rooms=config.model.num_rooms,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        pooling=config.model.pooling,
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create loss
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
    
    # Get data
    dataloaders = get_dataloaders(
        config.data,
        batch_size=config.training.batch_size,
        num_workers=0,
    )
    
    batch = next(iter(dataloaders["train"]))
    
    # Move to device
    X = batch["X"].to(device)
    Y = batch["Y"].to(device)
    context = batch["context"].to(device)
    context_mask = batch["context_mask"].to(device)
    
    # Forward pass
    model.train()
    logits = model(X, context, context_mask)
    loss = loss_fn(logits, Y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X, context, context_mask)
        probs = torch.sigmoid(logits)
    
    metrics = compute_metrics(probs.cpu(), Y.cpu(), threshold=0.5)
    
    print(f"Device: {device}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"✓ Training step test passed!\n")


def main():
    print("\n" + "=" * 80)
    print("RUNNING PIPELINE TESTS")
    print("=" * 80 + "\n")
    
    try:
        test_model()
        test_data_loading()
        test_loss()
        test_training_step()
        
        print("=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nReady to train! Run:")
        print("  python -m deepset.train --config debug")
        print("\nOr for full training:")
        print("  python -m deepset.train --config default")
        print("\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
