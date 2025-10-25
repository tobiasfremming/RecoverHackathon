"""
Evaluation Utilities

Functions for evaluating model performance and analyzing predictions.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics.score import normalized_rooms_score, get_room_scores


def apply_threshold(
    predictions: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Apply threshold to predictions to get binary outputs.
    
    Args:
        predictions: (batch, num_clusters) - sigmoid probabilities
        threshold: Threshold value
    
    Returns:
        binary_predictions: (batch, num_clusters) - binary predictions
    """
    return (predictions >= threshold).float()


def predictions_to_operation_codes(
    predictions: torch.Tensor,
    threshold: float = 0.5
) -> List[List[int]]:
    """
    Convert predictions to lists of operation codes.
    
    Args:
        predictions: (batch, num_clusters) - sigmoid probabilities
        threshold: Threshold value
    
    Returns:
        operation_codes: List of lists of predicted operation codes
    """
    binary_preds = apply_threshold(predictions, threshold)
    codes = []
    
    for i in range(binary_preds.shape[0]):
        sample_codes = torch.where(binary_preds[i] > 0.5)[0].tolist()
        codes.append(sample_codes)
    
    return codes


def evaluate_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate predictions against targets.
    
    Args:
        predictions: (batch, num_clusters) - sigmoid probabilities
        targets: (batch, num_clusters) - binary targets
        threshold: Threshold for binary predictions
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Convert to binary predictions
    binary_preds = apply_threshold(predictions, threshold)
    
    # Convert to numpy for easier computation
    preds_np = binary_preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Compute per-sample metrics
    tp = ((preds_np == 1) & (targets_np == 1)).sum(axis=1)
    fp = ((preds_np == 1) & (targets_np == 0)).sum(axis=1)
    fn = ((preds_np == 0) & (targets_np == 1)).sum(axis=1)
    tn = ((preds_np == 0) & (targets_np == 0)).sum(axis=1)
    
    # Aggregate metrics
    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    total_tn = tn.sum()
    
    # Precision, Recall, F1
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Accuracy
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-8)
    
    # Hamming loss (fraction of wrong labels)
    hamming = (total_fp + total_fn) / (preds_np.size)
    
    # Exact match ratio (fraction of samples with all labels correct)
    exact_match = (preds_np == targets_np).all(axis=1).mean()
    
    # Compute custom room score
    pred_codes = predictions_to_operation_codes(predictions, threshold)
    target_codes = predictions_to_operation_codes(targets, threshold=0.5)
    room_score = normalized_rooms_score(pred_codes, target_codes)
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "hamming_loss": hamming,
        "exact_match": exact_match,
        "room_score": room_score,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn
    }
    
    return metrics


def compute_per_operation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[int, Dict[str, float]]:
    """
    Compute precision, recall, F1 for each operation separately.
    
    Args:
        predictions: (batch, num_clusters) - sigmoid probabilities
        targets: (batch, num_clusters) - binary targets
        threshold: Threshold for binary predictions
    
    Returns:
        per_op_metrics: Dict mapping operation code to metrics dict
    """
    binary_preds = apply_threshold(predictions, threshold)
    
    preds_np = binary_preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    num_clusters = preds_np.shape[1]
    per_op_metrics = {}
    
    for op_code in range(num_clusters):
        pred_col = preds_np[:, op_code]
        target_col = targets_np[:, op_code]
        
        tp = ((pred_col == 1) & (target_col == 1)).sum()
        fp = ((pred_col == 1) & (target_col == 0)).sum()
        fn = ((pred_col == 0) & (target_col == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        support = target_col.sum()
        
        per_op_metrics[op_code] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    return per_op_metrics


def find_optimal_threshold(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric: str = "room_score",
    search_range: Tuple[float, float] = (0.1, 0.9),
    num_steps: int = 41
) -> Tuple[float, float]:
    """
    Find optimal threshold by grid search.
    
    Args:
        predictions: (batch, num_clusters) - sigmoid probabilities
        targets: (batch, num_clusters) - binary targets
        metric: Metric to optimize ("room_score", "f1", "precision", "recall")
        search_range: (min, max) threshold range
        num_steps: Number of thresholds to try
    
    Returns:
        best_threshold: Optimal threshold value
        best_score: Score achieved at optimal threshold
    """
    thresholds = np.linspace(search_range[0], search_range[1], num_steps)
    best_threshold = 0.5
    best_score = -float('inf')
    
    for threshold in thresholds:
        metrics = evaluate_predictions(predictions, targets, threshold)
        score = metrics[metric]
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def analyze_errors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    top_k: int = 20
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Analyze error patterns.
    
    Args:
        predictions: (batch, num_clusters) - sigmoid probabilities
        targets: (batch, num_clusters) - binary targets
        threshold: Threshold for binary predictions
        top_k: Number of top errors to return
    
    Returns:
        error_analysis: Dict with most common FP and FN operations
    """
    binary_preds = apply_threshold(predictions, threshold)
    
    preds_np = binary_preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    num_clusters = preds_np.shape[1]
    
    # Count FP and FN for each operation
    fp_counts = {}
    fn_counts = {}
    
    for op_code in range(num_clusters):
        pred_col = preds_np[:, op_code]
        target_col = targets_np[:, op_code]
        
        fp = ((pred_col == 1) & (target_col == 0)).sum()
        fn = ((pred_col == 0) & (target_col == 1)).sum()
        
        if fp > 0:
            fp_counts[op_code] = fp
        if fn > 0:
            fn_counts[op_code] = fn
    
    # Sort by count
    top_fp = sorted(fp_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_fn = sorted(fn_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return {
        "top_false_positives": top_fp,
        "top_false_negatives": top_fn
    }


def evaluate_by_room_cluster(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    room_clusters: torch.Tensor,
    threshold: float = 0.5,
    num_rooms: int = 11
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate performance separately for each room cluster.
    
    Args:
        predictions: (batch, num_clusters) - sigmoid probabilities
        targets: (batch, num_clusters) - binary targets
        room_clusters: (batch,) - room cluster indices
        threshold: Threshold for binary predictions
        num_rooms: Number of room clusters
    
    Returns:
        per_room_metrics: Dict mapping room cluster to metrics
    """
    per_room_metrics = {}
    
    room_clusters_np = room_clusters.cpu().numpy()
    
    for room_idx in range(num_rooms):
        # Find samples from this room cluster
        mask = room_clusters_np == room_idx
        
        if mask.sum() == 0:
            continue
        
        # Evaluate on this subset
        room_preds = predictions[mask]
        room_targets = targets[mask]
        
        metrics = evaluate_predictions(room_preds, room_targets, threshold)
        metrics["num_samples"] = mask.sum()
        
        per_room_metrics[room_idx] = metrics
    
    return per_room_metrics


class MetricsTracker:
    """Track metrics over training."""
    
    def __init__(self):
        self.history = defaultdict(list)
        self.epoch_metrics = {}
    
    def update(self, metrics: Dict[str, float], prefix: str = ""):
        """Add metrics for current step."""
        for key, value in metrics.items():
            full_key = f"{prefix}{key}" if prefix else key
            self.history[full_key].append(value)
    
    def get_epoch_summary(self, prefix: str = "") -> Dict[str, float]:
        """Get summary of metrics for current epoch."""
        summary = {}
        for key in self.history.keys():
            if key.startswith(prefix):
                values = self.history[key]
                if len(values) > 0:
                    summary[key] = np.mean(values)
        return summary
    
    def reset_epoch(self):
        """Reset metrics for new epoch."""
        self.history.clear()
    
    def save(self, path: str):
        """Save metrics history to file."""
        import json
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
    
    def load(self, path: str):
        """Load metrics history from file."""
        import json
        with open(path, 'r') as f:
            self.history = defaultdict(list, json.load(f))


if __name__ == "__main__":
    # Test evaluation utilities
    print("Testing evaluation utilities...")
    
    batch_size = 10
    num_clusters = 20
    
    # Create dummy data
    torch.manual_seed(42)
    predictions = torch.sigmoid(torch.randn(batch_size, num_clusters))
    targets = torch.randint(0, 2, (batch_size, num_clusters)).float()
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Test basic evaluation
    print("\n1. Basic Evaluation:")
    metrics = evaluate_predictions(predictions, targets, threshold=0.5)
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    
    # Test per-operation metrics
    print("\n2. Per-Operation Metrics (showing first 5):")
    per_op = compute_per_operation_metrics(predictions, targets)
    for op_code in list(per_op.keys())[:5]:
        print(f"   Op {op_code}: F1={per_op[op_code]['f1']:.4f}, "
              f"Support={per_op[op_code]['support']}")
    
    # Test threshold tuning
    print("\n3. Threshold Tuning:")
    best_thresh, best_score = find_optimal_threshold(
        predictions, targets, metric="f1", num_steps=21
    )
    print(f"   Best threshold: {best_thresh:.3f}")
    print(f"   Best F1 score: {best_score:.4f}")
    
    # Test error analysis
    print("\n4. Error Analysis:")
    errors = analyze_errors(predictions, targets, top_k=5)
    print(f"   Top 5 False Positives: {errors['top_false_positives']}")
    print(f"   Top 5 False Negatives: {errors['top_false_negatives']}")
    
    # Test metrics tracker
    print("\n5. Metrics Tracker:")
    tracker = MetricsTracker()
    tracker.update({"loss": 0.5, "f1": 0.7}, prefix="train_")
    tracker.update({"loss": 0.4, "f1": 0.75}, prefix="train_")
    summary = tracker.get_epoch_summary(prefix="train_")
    print(f"   Epoch summary: {summary}")
    
    print("\nAll tests passed!")
