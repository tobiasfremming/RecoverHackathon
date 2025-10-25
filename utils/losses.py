"""
Custom Loss Functions for Deep Sets Autoencoder

Implements weighted BCE loss and Focal Loss for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss with class weights.
    
    Weights rare operations more heavily to handle class imbalance.
    Uses BCEWithLogitsLoss for numerical stability.
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_clusters) - raw logits (before sigmoid)
            targets: (batch, num_clusters) - binary targets
        
        Returns:
            loss: Scalar loss value
        """
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss focuses on hard examples by down-weighting easy examples.
    Formula: FL = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where p_t is the probability of the true class.
    """
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_clusters) - raw logits (before sigmoid)
            targets: (batch, num_clusters) - binary targets
        
        Returns:
            loss: Scalar loss value
        """
        # Apply sigmoid to get probabilities for focal weight calculation
        predictions = torch.sigmoid(logits)
        
        # Compute binary cross-entropy with logits (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )
        
        # Compute p_t (probability of true class)
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha_t (class balancing)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        # Apply additional positive class weights if provided
        if self.pos_weight is not None:
            weight = torch.ones_like(targets)
            weight = weight + (self.pos_weight - 1) * targets
            focal_loss = focal_loss * weight
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for main task (operation prediction) and auxiliary task (room complete).
    """
    
    def __init__(
        self,
        main_loss: nn.Module,
        main_weight: float = 10.0,
        auxiliary_weight: float = 0.05,
        use_auxiliary: bool = True
    ):
        super().__init__()
        self.main_loss = main_loss
        self.main_weight = main_weight
        self.auxiliary_weight = auxiliary_weight
        self.use_auxiliary = use_auxiliary
        self.auxiliary_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        complete_logit: Optional[torch.Tensor] = None,
        complete_target: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Args:
            logits: (batch, num_clusters) - raw logits for operations
            targets: (batch, num_clusters) - operation targets
            complete_logit: (batch, 1) - raw logit for room completeness (optional)
            complete_target: (batch, 1) - room complete targets (optional)
        
        Returns:
            total_loss: Combined loss
            main_loss_value: Main task loss (for logging)
            aux_loss_value: Auxiliary task loss (for logging)
        """
        # Main loss (weighted)
        main_loss_value = self.main_loss(logits, targets)
        total_loss = self.main_weight * main_loss_value
        
        # Auxiliary loss (weighted)
        aux_loss_value = torch.tensor(0.0, device=logits.device)
        if self.use_auxiliary and complete_logit is not None and complete_target is not None:
            aux_loss_value = self.auxiliary_loss(complete_logit, complete_target)
            total_loss = total_loss + self.auxiliary_weight * aux_loss_value
        
        return total_loss, main_loss_value, aux_loss_value


def compute_room_complete_targets(Y: torch.Tensor) -> torch.Tensor:
    """
    Compute 'room complete' targets from Y.
    
    A room is complete if there are no masked operations (Y is all zeros).
    
    Args:
        Y: (batch, num_clusters) - target operations (1 for masked ops)
    
    Returns:
        complete_targets: (batch, 1) - 1 if room is complete, 0 otherwise
    """
    # Room is complete if sum of Y is 0
    has_missing_ops = (Y.sum(dim=1) > 0).float()
    complete_targets = 1 - has_missing_ops
    return complete_targets.unsqueeze(-1)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Applies different focusing parameters for positive and negative samples.
    Useful when false negatives are more costly than false positives.
    """
    
    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_clusters) - raw logits (before sigmoid)
            targets: (batch, num_clusters) - binary targets
        
        Returns:
            loss: Scalar loss value
        """
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(logits)
        
        # Clip predictions for numerical stability
        predictions = torch.clamp(predictions, min=self.clip, max=1.0 - self.clip)
        
        # Positive loss
        pos_loss = -targets * torch.log(predictions)
        pos_loss = pos_loss * (1 - predictions) ** self.gamma_pos
        
        # Negative loss
        neg_loss = -(1 - targets) * torch.log(1 - predictions)
        neg_loss = neg_loss * predictions ** self.gamma_neg
        
        # Total loss
        loss = pos_loss + neg_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def create_loss_function(
    loss_config,
    class_weights: Optional[torch.Tensor] = None,
    device: str = "cuda"
):
    """
    Create loss function based on configuration.
    
    Args:
        loss_config: LossConfig object
        class_weights: Optional class weights tensor
        device: Device to place tensors on
    
    Returns:
        loss_fn: Configured loss function
    """
    # Prepare pos_weight
    pos_weight = None
    if loss_config.use_class_weights and class_weights is not None:
        pos_weight = class_weights.to(device) * loss_config.pos_weight_scale
    
    # Create main loss
    if loss_config.use_focal_loss:
        print(f"Using Focal Loss (alpha={loss_config.focal_alpha}, gamma={loss_config.focal_gamma})")
        main_loss = FocalLoss(
            alpha=loss_config.focal_alpha,
            gamma=loss_config.focal_gamma,
            reduction=loss_config.bce_reduction,
            pos_weight=pos_weight
        )
    else:
        print("Using Weighted BCE Loss")
        main_loss = WeightedBCELoss(
            pos_weight=pos_weight,
            reduction=loss_config.bce_reduction
        )
    
    # Get loss weights from config or use defaults
    main_weight = getattr(loss_config, 'main_loss_weight', 10.0)
    aux_weight = getattr(loss_config, 'auxiliary_loss_weight', 0.05)
    
    print(f"Loss weights: main={main_weight}, auxiliary={aux_weight}")
    
    # Wrap in combined loss
    loss_fn = CombinedLoss(
        main_loss=main_loss,
        main_weight=main_weight,
        auxiliary_weight=aux_weight,
        use_auxiliary=True
    )
    
    return loss_fn


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    batch_size = 4
    num_clusters = 10
    
    # Create dummy data
    predictions = torch.sigmoid(torch.randn(batch_size, num_clusters))
    targets = torch.randint(0, 2, (batch_size, num_clusters)).float()
    complete_pred = torch.sigmoid(torch.randn(batch_size, 1))
    complete_target = compute_room_complete_targets(targets)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Complete targets: {complete_target.squeeze().tolist()}")
    
    # Test WeightedBCELoss
    print("\n1. Weighted BCE Loss:")
    weights = torch.rand(num_clusters) + 0.5  # Random weights
    bce_loss = WeightedBCELoss(pos_weight=weights)
    loss_val = bce_loss(predictions, targets)
    print(f"   Loss value: {loss_val.item():.4f}")
    
    # Test FocalLoss
    print("\n2. Focal Loss:")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_val = focal_loss(predictions, targets)
    print(f"   Loss value: {loss_val.item():.4f}")
    
    # Test AsymmetricLoss
    print("\n3. Asymmetric Loss:")
    asym_loss = AsymmetricLoss(gamma_pos=1.0, gamma_neg=4.0)
    loss_val = asym_loss(predictions, targets)
    print(f"   Loss value: {loss_val.item():.4f}")
    
    # Test CombinedLoss
    print("\n4. Combined Loss:")
    main_loss = WeightedBCELoss(pos_weight=weights)
    combined_loss = CombinedLoss(main_loss, auxiliary_weight=0.1)
    total, main_val, aux_val = combined_loss(
        predictions, targets, complete_pred, complete_target
    )
    print(f"   Total loss: {total.item():.4f}")
    print(f"   Main loss: {main_val.item():.4f}")
    print(f"   Auxiliary loss: {aux_val.item():.4f}")
    
    print("\nAll tests passed!")
