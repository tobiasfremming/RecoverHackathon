"""
Inference Script for Deep Sets Autoencoder

Generate predictions on test set and create submission file.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.hackathon import HackathonDataset
from dataset.collate import collate_fn
from models.deep_sets_autoencoder import DeepSetsAutoencoder
from config.train_config import ExperimentConfig, get_default_config
from utils.features import FeatureNormalizer
from utils.evaluation import (
    predictions_to_operation_codes,
    find_optimal_threshold
)


def prepare_batch(batch: Dict, device: str) -> Dict:
    """
    Prepare batch for model input.
    
    Same as in train.py - converts collate_fn output to model-ready format.
    """
    # Extract X (operations + room_cluster concatenated)
    X_full = batch["X"]  # (batch, num_clusters + num_rooms)
    num_clusters = 388
    num_rooms = 11
    
    # Split X into operations and room_cluster
    X_operations = X_full[:, :num_clusters]  # (batch, num_clusters)
    room_cluster_one_hot = X_full[:, num_clusters:]  # (batch, num_rooms)
    
    # Convert X operations from one-hot to codes
    batch_size = X_operations.shape[0]
    max_ops = 50
    
    X_codes_list = []
    X_mask_list = []
    
    for i in range(batch_size):
        codes = torch.where(X_operations[i] > 0.5)[0]
        num_codes = min(len(codes), max_ops)
        
        padded_codes = torch.zeros(max_ops, dtype=torch.long)
        mask = torch.zeros(max_ops, dtype=torch.bool)
        
        if num_codes > 0:
            padded_codes[:num_codes] = codes[:num_codes]
            mask[:num_codes] = True
        
        X_codes_list.append(padded_codes)
        X_mask_list.append(mask)
    
    X_codes = torch.stack(X_codes_list).to(device)
    X_mask = torch.stack(X_mask_list).to(device)
    
    context = batch["context"].to(device)
    context_mask = batch["context_mask"].to(device)
    Y = batch["Y"].float().to(device) if "Y" in batch else None  # Ensure Y is float
    
    return {
        "X_codes": X_codes,
        "X_mask": X_mask,
        "context": context,
        "context_mask": context_mask,
        "room_cluster_one_hot": room_cluster_one_hot.to(device),
        "Y": Y
    }


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    metadata_dataset,
    normalizer: FeatureNormalizer
) -> torch.Tensor:
    """
    Run inference on dataset.
    
    Returns:
        predictions: Tensor of shape (total_samples, num_clusters) with probabilities
    """
    model.eval()
    
    all_predictions = []
    
    for batch in tqdm(dataloader, desc="Running inference"):
        # Prepare batch
        prepared = prepare_batch(batch, device)
        
        # Get metadata (dummy for now - replace with actual metadata)
        batch_size = prepared["X_codes"].shape[0]
        insurance_company_one_hot = torch.zeros(batch_size, 14).to(device)
        insurance_company_one_hot[:, 0] = 1
        office_distance = torch.zeros(batch_size, 1).to(device)
        case_creation_year = torch.zeros(batch_size, 1).to(device)
        case_creation_month = torch.ones(batch_size, 1).to(device) * 6
        
        # Forward pass
        logits, _ = model(
            X_codes=prepared["X_codes"],
            X_mask=prepared["X_mask"],
            context=prepared["context"],
            context_mask=prepared["context_mask"],
            insurance_company_one_hot=insurance_company_one_hot,
            room_cluster_one_hot=prepared["room_cluster_one_hot"],
            office_distance=office_distance,
            case_creation_year=case_creation_year,
            case_creation_month=case_creation_month
        )
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(logits)
        
        all_predictions.append(predictions.cpu())
    
    return torch.cat(all_predictions, dim=0)


def tune_threshold_on_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    metadata_dataset,
    normalizer: FeatureNormalizer,
    metric: str = "room_score"
) -> float:
    """
    Tune threshold on validation set.
    
    Returns:
        best_threshold: Optimal threshold value
    """
    print("Tuning threshold on validation set...")
    
    # Run inference on validation set
    val_predictions = run_inference(
        model, val_loader, device, metadata_dataset, normalizer
    )
    
    # Get validation targets
    all_targets = []
    for batch in val_loader:
        all_targets.append(batch["Y"])
    val_targets = torch.cat(all_targets, dim=0)
    
    # Find optimal threshold
    from utils.evaluation import find_optimal_threshold
    best_threshold, best_score = find_optimal_threshold(
        val_predictions,
        val_targets,
        metric=metric,
        search_range=(0.1, 0.9),
        num_steps=41
    )
    
    print(f"Best threshold: {best_threshold:.3f} (score: {best_score:.4f})")
    
    return best_threshold


def create_submission(
    dataset: HackathonDataset,
    predictions: torch.Tensor,
    threshold: float,
    output_dir: str = "submissions"
) -> str:
    """
    Create submission file from predictions.
    
    Args:
        dataset: HackathonDataset for test split
        predictions: Tensor of shape (num_samples, num_clusters)
        threshold: Threshold for binary predictions
        output_dir: Directory to save submission
    
    Returns:
        submission_path: Path to saved submission file
    """
    # Convert predictions to operation codes
    pred_codes = predictions_to_operation_codes(predictions, threshold)
    
    # Get IDs from dataset
    # Note: We need to access the work_operations_dataset to get IDs
    # For now, create a mapping using indices
    predictions_dict = {}
    
    for idx in range(len(dataset)):
        sample = dataset.work_operations_dataset[idx]
        sample_id = sample["id"]
        predictions_dict[sample_id] = pred_codes[idx]
    
    # Create submission using dataset method
    dataset.create_submission(predictions_dict)
    
    # The create_submission method saves to submissions/ directory
    # Return the path (you may need to track the timestamp)
    import pandas as pd
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"submissions/submission_{timestamp}.csv"
    
    return submission_path


def load_model(
    checkpoint_path: str,
    config: ExperimentConfig,
    device: str
) -> nn.Module:
    """Load model from checkpoint."""
    
    model = DeepSetsAutoencoder(
        num_clusters=config.model.num_clusters,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        num_companies=config.model.num_companies,
        num_rooms=config.model.num_rooms,
        pooling_type=config.model.pooling_type,
        use_attention=config.model.use_attention,
        dropout=config.model.dropout,
        use_auxiliary_head=config.model.use_auxiliary_head
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if "metrics" in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    return model


def main(args):
    """Main inference function."""
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = get_default_config()
    config_path = Path(args.checkpoint).parent / "config.json"
    if config_path.exists():
        print(f"Loading config from {config_path}")
        # Note: Full config loading would require more complex deserialization
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    
    # Load datasets
    print("Loading datasets...")
    
    # Set num_workers=0 on Windows or CPU to avoid multiprocessing serialization issues
    num_workers = 0 if (os.name == 'nt' or device.type == 'cpu') else 4
    use_pin_memory = device.type == 'cuda'
    
    # Validation set (for threshold tuning)
    val_dataset = None
    val_loader = None
    if args.tune_threshold:
        val_dataset = HackathonDataset(
            root=args.data_root,
            split="val",
            download=False,
            seed=42
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        )
        print(f"Validation dataset size: {len(val_dataset)}")
    
    # Test set
    test_dataset = HackathonDataset(
        root=args.data_root,
        split="test",
        download=False,
        seed=42
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create feature normalizer (load from training if available)
    normalizer = FeatureNormalizer()
    
    # Tune threshold on validation set
    threshold = args.threshold
    if args.tune_threshold and val_loader:
        threshold = tune_threshold_on_validation(
            model=model,
            val_loader=val_loader,
            device=device,
            metadata_dataset=val_dataset.metadata_dataset,
            normalizer=normalizer,
            metric="room_score"
        )
    else:
        print(f"Using threshold: {threshold}")
    
    # Run inference on test set
    print("\nRunning inference on test set...")
    test_predictions = run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
        metadata_dataset=test_dataset.metadata_dataset,
        normalizer=normalizer
    )
    
    print(f"Predictions shape: {test_predictions.shape}")
    print(f"Predictions range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
    
    # Create submission
    print("\nCreating submission file...")
    submission_path = create_submission(
        dataset=test_dataset,
        predictions=test_predictions,
        threshold=threshold,
        output_dir=args.output_dir
    )
    
    print(f"✓ Submission saved to: {submission_path}")
    
    # Save predictions
    if args.save_predictions:
        pred_path = Path(args.output_dir) / "test_predictions.pt"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "predictions": test_predictions,
            "threshold": threshold
        }, pred_path)
        print(f"✓ Raw predictions saved to: {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Deep Sets Autoencoder")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="data",
                       help="Root directory for data")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Default threshold for predictions")
    parser.add_argument("--tune_threshold", action="store_true",
                       help="Tune threshold on validation set")
    parser.add_argument("--output_dir", type=str, default="submissions",
                       help="Output directory for submission")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save raw predictions tensor")
    
    args = parser.parse_args()
    
    main(args)
