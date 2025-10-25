"""Utility functions for feature engineering, losses, and evaluation."""

from .features import (
    normalize_numeric_features,
    cyclical_encode_month,
    compute_class_weights,
    extract_operation_codes_from_one_hot,
    codes_to_padded_tensor,
    FeatureNormalizer
)

from .losses import (
    WeightedBCELoss,
    FocalLoss,
    CombinedLoss,
    AsymmetricLoss,
    compute_room_complete_targets,
    create_loss_function
)

from .evaluation import (
    apply_threshold,
    predictions_to_operation_codes,
    evaluate_predictions,
    compute_per_operation_metrics,
    find_optimal_threshold,
    analyze_errors,
    evaluate_by_room_cluster,
    MetricsTracker
)

__all__ = [
    # Features
    "normalize_numeric_features",
    "cyclical_encode_month",
    "compute_class_weights",
    "extract_operation_codes_from_one_hot",
    "codes_to_padded_tensor",
    "FeatureNormalizer",
    # Losses
    "WeightedBCELoss",
    "FocalLoss",
    "CombinedLoss",
    "AsymmetricLoss",
    "compute_room_complete_targets",
    "create_loss_function",
    # Evaluation
    "apply_threshold",
    "predictions_to_operation_codes",
    "evaluate_predictions",
    "compute_per_operation_metrics",
    "find_optimal_threshold",
    "analyze_errors",
    "evaluate_by_room_cluster",
    "MetricsTracker"
]
