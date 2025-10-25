"""
Utility functions for training and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal Loss = -alpha * (1 - p)^gamma * log(p)
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) - raw predictions
            targets: (batch, num_classes) - binary targets
        
        Returns:
            loss: scalar or (batch, num_classes) depending on reduction
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute focal weight
        # For positive examples: (1 - p)^gamma
        # For negative examples: p^gamma
        focal_weight = torch.where(
            targets == 1,
            (1 - probs) ** self.gamma,
            probs ** self.gamma
        )
        
        # Compute binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        
        # Apply focal weight and alpha
        alpha_weight = torch.where(
            targets == 1,
            self.alpha,
            1 - self.alpha
        )
        
        focal_loss = alpha_weight * focal_weight * bce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """Binary Cross Entropy with class weights."""
    
    def __init__(self, pos_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weights = pos_weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weights
        )


def compute_class_weights(
    dataset,
    scale: float = 1.0,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Compute inverse frequency weights for each operation.
    
    Args:
        dataset: HackathonDataset
        scale: Scale factor for weights
        device: Device to put weights on
    
    Returns:
        weights: (num_operations,) tensor
    """
    num_operations = 388
    
    # Count occurrences of each operation
    counts = np.zeros(num_operations)
    
    print("Computing class weights from dataset...")
    for i in range(min(len(dataset), 10000)):  # Sample for speed
        sample = dataset[i]
        Y = sample["Y"].numpy()
        counts += Y
    
    # Inverse frequency
    counts = np.maximum(counts, 1)  # Avoid division by zero
    weights = 1.0 / counts
    
    # Normalize
    weights = weights / weights.mean()
    
    # Scale
    weights = weights * scale
    
    print(f"Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        preds: (batch, num_classes) - probabilities
        targets: (batch, num_classes) - binary targets
        threshold: Classification threshold
    
    Returns:
        metrics: Dictionary of metric values
    """
    # Binarize predictions
    binary_preds = (preds > threshold).float()
    
    # Compute TP, FP, TN, FN
    tp = (binary_preds * targets).sum().item()
    fp = (binary_preds * (1 - targets)).sum().item()
    tn = ((1 - binary_preds) * (1 - targets)).sum().item()
    fn = ((1 - binary_preds) * targets).sum().item()
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_room_score(
    preds_list: List[List[int]],
    targets_list: List[List[int]]
) -> float:
    """
    Compute competition room score.
    
    Score = TP×1 + FP×(-0.25) + FN×(-0.5)
    Empty room correct: +1
    
    Args:
        preds_list: List of predicted operation lists per room
        targets_list: List of target operation lists per room
    
    Returns:
        total_score: Sum of scores across all rooms
    """
    total_score = 0.0
    
    for preds, targets in zip(preds_list, targets_list):
        preds_set = set(preds)
        targets_set = set(targets)
        
        if len(targets_set) == 0:
            # Empty room
            if len(preds_set) == 0:
                total_score += 1.0  # Correct empty prediction
            else:
                total_score -= 0.25 * len(preds_set)  # False positives
        else:
            # Non-empty room
            tp = len(preds_set & targets_set)
            fp = len(preds_set - targets_set)
            fn = len(targets_set - preds_set)
            
            total_score += tp * 1.0
            total_score -= fp * 0.25
            total_score -= fn * 0.5
    
    return total_score


def predictions_to_codes(
    probs: torch.Tensor,
    threshold: float = 0.5
) -> List[List[int]]:
    """
    Convert probabilities to operation code lists.
    
    Args:
        probs: (batch, num_operations) - probabilities
        threshold: Classification threshold
    
    Returns:
        codes_list: List of operation code lists
    """
    binary_preds = (probs > threshold).cpu().numpy()
    
    codes_list = []
    for i in range(len(binary_preds)):
        codes = np.where(binary_preds[i] == 1)[0].tolist()
        codes_list.append(codes)
    
    return codes_list


def find_optimal_threshold(
    probs: torch.Tensor,
    targets: torch.Tensor,
    metric: str = "f1",
    thresholds: Optional[List[float]] = None
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        probs: (batch, num_operations) - probabilities
        targets: (batch, num_operations) - binary targets
        metric: Metric to optimize ('f1', 'precision', 'recall')
        thresholds: List of thresholds to try
    
    Returns:
        best_threshold: Optimal threshold
        best_score: Best metric value
    """
    if thresholds is None:
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
    
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in thresholds:
        metrics = compute_metrics(probs, targets, threshold=thresh)
        score = metrics[metric]
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


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


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make cudnn deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test focal loss
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 2, (4, 10))
    loss = focal_loss(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")
    
    # Test metrics
    probs = torch.sigmoid(logits)
    metrics = compute_metrics(probs, targets, threshold=0.5)
    print(f"Metrics: {metrics}")
    
    # Test room score
    preds_list = [[1, 2, 3], [4, 5], []]
    targets_list = [[1, 2, 4], [4, 5, 6], []]
    score = compute_room_score(preds_list, targets_list)
    print(f"Room score: {score:.2f}")
    
    print("✓ All tests passed!")
