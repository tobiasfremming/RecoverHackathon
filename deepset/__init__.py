"""
Deep Sets implementation for Recover Hackathon.

This package provides a clean, modular implementation of Deep Sets
for multi-label classification of work operations.
"""

from .model import DeepSetsModel
from .config import get_config, Config, ModelConfig, TrainingConfig, LossConfig, DataConfig
from .utils import (
    FocalLoss,
    WeightedBCELoss,
    compute_metrics,
    compute_room_score,
    set_seed,
)
from .data_loader import get_dataloaders

__version__ = "1.0.0"
__all__ = [
    "DeepSetsModel",
    "get_config",
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "LossConfig",
    "DataConfig",
    "FocalLoss",
    "WeightedBCELoss",
    "compute_metrics",
    "compute_room_score",
    "set_seed",
    "get_dataloaders",
]
