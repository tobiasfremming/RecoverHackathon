"""
Per-operation weight calculator for reducing FP rate on problematic operations.

Based on the performance analysis, certain operations have very high FP rates:
- Op 260: 84.1% FP rate
- Op 108: 80.9% FP rate
- Op 204: 80.0% FP rate
- Op 257: 75.7% FP rate
- Op 259: 80.6% FP rate

This script computes per-operation weights to discourage FP predictions.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def compute_per_operation_weights(
    problematic_ops: list[int],
    num_operations: int = 388,
    base_weight: float = 1.0,
    problematic_weight: float = 2.0,
) -> torch.Tensor:
    """
    Compute per-operation weights for loss function.
    
    Problematic operations get higher weights to penalize FPs more.
    
    Args:
        problematic_ops: List of operation IDs with high FP rates
        num_operations: Total number of operations
        base_weight: Weight for normal operations
        problematic_weight: Weight for problematic operations (higher = more penalty)
    
    Returns:
        weights: Tensor of shape (num_operations,)
    """
    weights = torch.ones(num_operations) * base_weight
    
    for op_id in problematic_ops:
        weights[op_id] = problematic_weight
    
    return weights


def get_problematic_operations_from_analysis() -> list[int]:
    """
    Return list of problematic operations from performance analysis.
    
    Based on deepset/performance_analysis.txt:
    - Top 10 operations with most negative room score contribution
    """
    # From analysis: Operations with high FP rates and negative scores
    problematic_ops = [
        260,  # Score: -96.25, FP rate: 84.1%
        108,  # Score: -94.50, FP rate: 80.9%
        204,  # Score: -89.50, FP rate: 80.0%
        257,  # Score: -72.75, FP rate: 75.7%
        259,  # Score: -72.25, FP rate: 80.6%
        154,  # Score: -71.75
        262,  # Score: -69.50
        258,  # Score: -69.25
        103,  # Score: -67.50
        112,  # Score: -67.25
    ]
    
    return problematic_ops


class PerOperationWeightedLoss(torch.nn.Module):
    """
    Binary cross-entropy loss with per-operation weights.
    
    Penalizes FPs more for operations that historically have high FP rates.
    """
    
    def __init__(
        self,
        operation_weights: torch.Tensor,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("operation_weights", operation_weights)
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            logits: (batch, num_operations)
            targets: (batch, num_operations)
        
        Returns:
            loss: Scalar
        """
        # Compute per-element BCE
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        
        # Apply per-operation weights
        weighted_bce = bce * self.operation_weights.unsqueeze(0)
        
        # Reduce
        if self.reduction == "mean":
            return weighted_bce.mean()
        elif self.reduction == "sum":
            return weighted_bce.sum()
        else:
            return weighted_bce


def test_weights():
    """Test per-operation weights."""
    print("Testing per-operation weights...\n")
    
    # Get problematic operations
    problematic_ops = get_problematic_operations_from_analysis()
    print(f"Problematic operations ({len(problematic_ops)}): {problematic_ops}\n")
    
    # Compute weights
    weights = compute_per_operation_weights(
        problematic_ops,
        num_operations=388,
        base_weight=1.0,
        problematic_weight=2.0,
    )
    
    print(f"Weight statistics:")
    print(f"  Min: {weights.min().item():.2f}")
    print(f"  Max: {weights.max().item():.2f}")
    print(f"  Mean: {weights.mean().item():.2f}")
    print(f"  Std: {weights.std().item():.2f}")
    
    # Show weights for some operations
    print(f"\nSample weights:")
    print(f"  Op 0 (normal): {weights[0].item():.2f}")
    print(f"  Op 260 (problematic): {weights[260].item():.2f}")
    print(f"  Op 108 (problematic): {weights[108].item():.2f}")
    
    # Test loss function
    print("\nTesting PerOperationWeightedLoss...")
    loss_fn = PerOperationWeightedLoss(weights)
    
    # Create dummy data
    batch_size = 4
    num_ops = 388
    logits = torch.randn(batch_size, num_ops)
    targets = torch.randint(0, 2, (batch_size, num_ops)).float()
    
    loss = loss_fn(logits, targets)
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_weights()
