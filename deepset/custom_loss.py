"""
Custom loss function aligned with competition scoring.

The competition scoring penalizes:
- False Negatives (FN): -0.5 points
- False Positives (FP): -0.25 points
- FN penalty is 2x FP penalty

This loss function directly optimizes for this asymmetric penalty structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompetitionAlignedLoss(nn.Module):
    """
    Loss function that matches competition scoring penalties.
    
    Standard BCE treats FP and FN equally.
    Focal loss can weight positive class more, but doesn't directly match
    the 2:1 FN:FP penalty ratio in the competition.
    
    This loss uses asymmetric weighting:
    - Weight for FN: 2.0 (matches -0.5 penalty)
    - Weight for FP: 1.0 (matches -0.25 penalty)
    
    Args:
        fn_weight: Weight for false negatives (default: 2.0)
        fp_weight: Weight for false positives (default: 1.0)
        class_weights: Optional per-operation weights (for rare operations)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        fn_weight: float = 2.0,
        fp_weight: float = 1.0,
        class_weights: torch.Tensor = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            logits: Model predictions (before sigmoid), shape (batch, num_ops)
            targets: Ground truth labels, shape (batch, num_ops)
        
        Returns:
            loss: Scalar loss value
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute base BCE loss for each element
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # Create asymmetric weights
        # When target=1: We want to heavily penalize if pred=0 (FN)
        # When target=0: We want to lightly penalize if pred=1 (FP)
        weights = torch.where(
            targets == 1,
            torch.full_like(targets, self.fn_weight),  # Weight for FN
            torch.full_like(targets, self.fp_weight)   # Weight for FP
        )
        
        # Apply weights
        weighted_bce = bce * weights
        
        # Apply class weights if provided
        if self.class_weights is not None:
            class_weights = self.class_weights.to(weighted_bce.device)
            weighted_bce = weighted_bce * class_weights
        
        # Reduction
        if self.reduction == 'mean':
            return weighted_bce.mean()
        elif self.reduction == 'sum':
            return weighted_bce.sum()
        else:
            return weighted_bce


class CompetitionFocalLoss(nn.Module):
    """
    Focal loss with asymmetric FN/FP weighting for competition scoring.
    
    Combines focal loss (focus on hard examples) with asymmetric weighting
    (FN penalty 2x FP penalty).
    
    Args:
        fn_weight: Weight for false negatives (default: 2.0)
        fp_weight: Weight for false positives (default: 1.0)
        gamma: Focal loss focusing parameter (default: 2.0)
        class_weights: Optional per-operation weights
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        fn_weight: float = 2.0,
        fp_weight: float = 1.0,
        gamma: float = 2.0,
        class_weights: torch.Tensor = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric focal loss.
        
        Args:
            logits: Model predictions (before sigmoid), shape (batch, num_ops)
            targets: Ground truth labels, shape (batch, num_ops)
        
        Returns:
            loss: Scalar loss value
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute base BCE
        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # Compute focal term: (1 - p_t)^gamma
        # p_t is the probability of the true class
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_term = (1 - p_t) ** self.gamma
        
        # Create asymmetric weights (FN vs FP)
        weights = torch.where(
            targets == 1,
            torch.full_like(targets, self.fn_weight),
            torch.full_like(targets, self.fp_weight)
        )
        
        # Combine: focal_loss = focal_term * weighted_bce
        focal_loss = focal_term * bce * weights
        
        # Apply class weights if provided
        if self.class_weights is not None:
            class_weights = self.class_weights.to(focal_loss.device)
            focal_loss = focal_loss * class_weights
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def test_losses():
    """Test the custom loss functions."""
    batch_size = 4
    num_ops = 10
    
    # Create dummy data
    logits = torch.randn(batch_size, num_ops)
    targets = torch.randint(0, 2, (batch_size, num_ops)).float()
    
    print("Testing Custom Loss Functions")
    print("=" * 60)
    
    # Test standard BCE
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    print(f"Standard BCE Loss: {bce_loss.item():.4f}")
    
    # Test competition-aligned loss
    comp_loss = CompetitionAlignedLoss(fn_weight=2.0, fp_weight=1.0)
    loss1 = comp_loss(logits, targets)
    print(f"Competition-Aligned Loss (FN=2.0, FP=1.0): {loss1.item():.4f}")
    
    # Test competition focal loss
    comp_focal = CompetitionFocalLoss(fn_weight=2.0, fp_weight=1.0, gamma=2.0)
    loss2 = comp_focal(logits, targets)
    print(f"Competition Focal Loss (FN=2.0, FP=1.0, γ=2.0): {loss2.item():.4f}")
    
    print("\nGradient check:")
    logits.requires_grad = True
    loss = comp_loss(logits, targets)
    loss.backward()
    print(f"Gradients computed successfully: {logits.grad is not None}")
    print(f"Gradient norm: {logits.grad.norm().item():.4f}")


if __name__ == "__main__":
    test_losses()
