"""
Training Script for Deep Sets Autoencoder

Main training loop with validation, checkpointing, and logging.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.hackathon import HackathonDataset
from dataset.collate import collate_fn
from models.deep_sets_autoencoder import DeepSetsAutoencoder
from config.train_config import ExperimentConfig, get_default_config
from utils.features import compute_class_weights, FeatureNormalizer
from utils.losses import create_loss_function, compute_room_complete_targets
from utils.evaluation import evaluate_predictions, MetricsTracker
from utils.model_health import ModelHealthMonitor
from utils.evaluation import (
    evaluate_predictions,
    find_optimal_threshold,
    analyze_errors,
    MetricsTracker
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_batch(batch: Dict, device: str) -> Dict:
    """
    Prepare batch for model input.
    
    Converts the collate_fn output to model-ready format.
    """
    # Extract X (operations + room_cluster concatenated)
    X_full = batch["X"]  # (batch, num_clusters + num_rooms)
    num_clusters = 388
    num_rooms = 11
    
    # Split X into operations and room_cluster
    X_operations = X_full[:, :num_clusters]  # (batch, num_clusters)
    room_cluster_one_hot = X_full[:, num_clusters:]  # (batch, num_rooms)
    
    # Convert X operations from one-hot to codes
    # For each sample, get indices where X_operations is 1
    batch_size = X_operations.shape[0]
    max_ops = 50  # Maximum number of operations per room
    
    X_codes_list = []
    X_mask_list = []
    
    for i in range(batch_size):
        codes = torch.where(X_operations[i] > 0.5)[0]
        num_codes = min(len(codes), max_ops)
        
        # Pad codes
        padded_codes = torch.zeros(max_ops, dtype=torch.long)
        mask = torch.zeros(max_ops, dtype=torch.bool)
        
        if num_codes > 0:
            # Ensure codes are within valid range [0, num_clusters-1]
            valid_codes = codes[codes < num_clusters]
            num_codes = min(len(valid_codes), max_ops)
            padded_codes[:num_codes] = valid_codes[:num_codes]
            mask[:num_codes] = True
        
        X_codes_list.append(padded_codes)
        X_mask_list.append(mask)
    
    X_codes = torch.stack(X_codes_list).to(device)
    X_mask = torch.stack(X_mask_list).to(device)
    
    # Extract other tensors
    context = batch["context"].to(device)
    context_mask = batch["context_mask"].to(device)
    Y = batch["Y"].float().to(device)  # Ensure Y is float for BCE loss
    
    return {
        "X_codes": X_codes,
        "X_mask": X_mask,
        "context": context,
        "context_mask": context_mask,
        "room_cluster_one_hot": room_cluster_one_hot.to(device),
        "Y": Y
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    grad_clip: float,
    metadata_dataset,
    tracker: MetricsTracker,
    normalizer: FeatureNormalizer,
    epoch: int,
    log_interval: int = 100
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    # Create health monitor
    health_monitor = ModelHealthMonitor(model)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    num_batches = 0
    
    for batch_idx, batch in enumerate(progress_bar):
        # Prepare batch
        prepared = prepare_batch(batch, device)
        
        # Get metadata for this batch
        # Note: We need project_ids from the original dataset
        # For simplicity, we'll use dummy metadata here
        # In production, modify collate_fn to include metadata
        batch_size = prepared["X_codes"].shape[0]
        
        # Dummy metadata (replace with actual metadata from dataset)
        insurance_company_one_hot = torch.zeros(batch_size, 14).to(device)
        insurance_company_one_hot[:, 0] = 1  # Default to company 0
        
        office_distance = torch.zeros(batch_size, 1).to(device)
        case_creation_year = torch.zeros(batch_size, 1).to(device)
        case_creation_month = torch.ones(batch_size, 1).to(device) * 6
        
        # Forward pass
        logits, complete_logit = model(
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
        
        # Check for NaN in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"\nWarning: NaN/Inf detected in logits at batch {batch_idx}")
            print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
            print(health_monitor.get_summary())
            # Reset optimizer state for this parameter group
            for param_group in optimizer.param_groups:
                for p in param_group['params']:
                    if p in optimizer.state:
                        optimizer.state[p] = {}
            continue
        
        # Compute targets
        Y = prepared["Y"]
        complete_target = compute_room_complete_targets(Y)
        
        # Compute loss
        total_loss, main_loss, aux_loss = loss_fn(
            logits, Y, complete_logit, complete_target
        )
        
        # Check for NaN in loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\nWarning: NaN/Inf loss at batch {batch_idx}")
            print(f"Main loss: {main_loss.item():.4f}, Aux loss: {aux_loss.item():.4f}")
            print(health_monitor.get_summary())
            continue
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Check gradients
        grad_stats = health_monitor.check_gradients()
        if grad_stats is None:
            print(f"\nSkipping batch {batch_idx} due to bad gradients")
            continue
        
        # Gradient clipping
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if grad_norm > grad_clip * 2:
                print(f"\nWarning: Large gradient norm at batch {batch_idx}: {grad_norm:.4f}")
        
        optimizer.step()
        num_batches += 1
        
        # Update metrics
        tracker.update({
            "loss": total_loss.item(),
            "main_loss": main_loss.item(),
            "aux_loss": aux_loss.item()
        }, prefix="train_")
        
        # Update progress bar and print health stats periodically
        if batch_idx % log_interval == 0:
            progress_bar.set_postfix({
                "loss": total_loss.item(),
                "main": main_loss.item(),
                "aux": aux_loss.item()
            })
            if batch_idx % (log_interval * 5) == 0 and batch_idx > 0:
                print(f"\n[Batch {batch_idx}] {health_monitor.get_summary()}")
    
    # Get epoch summary
    summary = tracker.get_epoch_summary(prefix="train_")
    return summary


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    metadata_dataset,
    normalizer: FeatureNormalizer,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Validate on validation set."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Validation"):
        # Prepare batch
        prepared = prepare_batch(batch, device)
        
        # Get metadata (dummy for now)
        batch_size = prepared["X_codes"].shape[0]
        insurance_company_one_hot = torch.zeros(batch_size, 14).to(device)
        insurance_company_one_hot[:, 0] = 1
        office_distance = torch.zeros(batch_size, 1).to(device)
        case_creation_year = torch.zeros(batch_size, 1).to(device)
        case_creation_month = torch.ones(batch_size, 1).to(device) * 6
        
        # Forward pass
        logits, complete_logit = model(
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
        
        # Compute loss
        Y = prepared["Y"]
        complete_target = compute_room_complete_targets(Y)
        
        batch_loss, main_loss, aux_loss = loss_fn(
            logits, Y, complete_logit, complete_target
        )
        
        total_loss += batch_loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item()
        
        # Apply sigmoid to logits for metric computation
        predictions = torch.sigmoid(logits)
        
        # Collect predictions and targets
        all_predictions.append(predictions)
        all_targets.append(Y)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Print prediction diagnostics
    print("\n" + "="*60)
    print("PREDICTION DIAGNOSTICS")
    print("="*60)
    
    # Logit statistics (before sigmoid)
    sample_logits = torch.cat([logits for _ in range(1)], dim=0)  # Last batch logits
    print(f"\nLogit Stats (last batch):")
    print(f"  Range: [{sample_logits.min():.4f}, {sample_logits.max():.4f}]")
    print(f"  Mean: {sample_logits.mean():.4f}, Std: {sample_logits.std():.4f}")
    
    # Probability statistics
    print(f"\nProbability Stats (all predictions):")
    print(f"  Range: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
    print(f"  Mean: {all_predictions.mean():.4f}, Std: {all_predictions.std():.4f}")
    print(f"  % Above 0.5: {(all_predictions > 0.5).float().mean().item() * 100:.2f}%")
    print(f"  % Above 0.3: {(all_predictions > 0.3).float().mean().item() * 100:.2f}%")
    print(f"  % Above 0.1: {(all_predictions > 0.1).float().mean().item() * 100:.2f}%")
    
    # Target statistics
    print(f"\nTarget Stats:")
    print(f"  Positive rate: {all_targets.mean().item() * 100:.2f}%")
    print(f"  Total positives: {all_targets.sum().item():.0f}")
    print(f"  Total negatives: {(1 - all_targets).sum().item():.0f}")
    
    # Prediction statistics (at threshold 0.5)
    binary_preds = (all_predictions > threshold).float()
    print(f"\nPrediction Stats (threshold={threshold}):")
    print(f"  Predicted positive rate: {binary_preds.mean().item() * 100:.2f}%")
    print(f"  Total predicted positives: {binary_preds.sum().item():.0f}")
    
    # Threshold Analysis
    print(f"\nThreshold Analysis:")
    best_f1 = 0
    best_thresh = threshold
    best_room_score = -float('inf')
    best_thresh_room = threshold
    
    for thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        preds_at_thresh = (all_predictions > thresh).float()
        tp = (preds_at_thresh * all_targets).sum().item()
        fp = (preds_at_thresh * (1 - all_targets)).sum().item()
        fn = ((1 - preds_at_thresh) * all_targets).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Compute room score (TP=+1, FP=-0.25, FN=-0.5)
        room_score = tp - 0.25 * fp - 0.5 * fn
        
        print(f"  @ {thresh}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, RoomScore={room_score:.1f}, Predicted={preds_at_thresh.sum().item():.0f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
        if room_score > best_room_score:
            best_room_score = room_score
            best_thresh_room = thresh
    
    print(f"\n  ⭐ Best F1: {best_f1:.4f} at threshold {best_thresh}")
    print(f"  ⭐ Best Room Score: {best_room_score:.1f} at threshold {best_thresh_room}")
    print("="*60 + "\n")
    
    # Use best threshold for metrics
    # If room score is negative, optimize for F1 instead (not doing well yet)
    # If room score is positive, use it (we're actually making progress)
    if best_room_score > 0:
        optimal_threshold = best_thresh_room
        print(f"Using optimal threshold (room score): {optimal_threshold}")
    else:
        optimal_threshold = best_thresh
        print(f"Using optimal threshold (F1): {optimal_threshold} (room scores all negative)")
    
    # Compute metrics with optimal threshold
    metrics = evaluate_predictions(all_predictions, all_targets, optimal_threshold)
    
    # Add loss metrics
    num_batches = len(dataloader)
    metrics["val_loss"] = total_loss / num_batches
    metrics["val_main_loss"] = total_main_loss / num_batches
    metrics["val_aux_loss"] = total_aux_loss / num_batches
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    config: ExperimentConfig,
    checkpoint_dir: str,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config
    }
    
    # Save latest checkpoint
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
    
    # Save periodic checkpoint
    if epoch % config.training.num_epochs == 0 or epoch % config.save_interval == 0:
        epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler = None
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


def main(config: ExperimentConfig):
    """Main training function."""
    
    # Set random seed
    set_seed(config.seed)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = Path(config.checkpoint_dir)
    log_dir = Path(config.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = checkpoint_dir / "config.json"
    with open(config_path, 'w') as f:
        # Convert dataclass to dict (simplified)
        json.dump({"seed": config.seed, "device": config.device}, f, indent=2)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = HackathonDataset(
        root=config.data_root,
        split="train",
        download=False,
        sampling_strategy=config.sampling.train_sampling_strategies,
        seed=config.seed
    )
    
    val_dataset = HackathonDataset(
        root=config.data_root,
        split="val",
        download=False,
        sampling_strategy=config.sampling.val_sampling_strategies,
        seed=config.seed
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    # Set num_workers=0 on Windows or CPU to avoid multiprocessing serialization issues
    num_workers = 0 if (os.name == 'nt' or device.type == 'cpu') else config.sampling.num_workers
    use_pin_memory = config.sampling.pin_memory and device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    # Create model
    print("Creating model...")
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create loss function
    print("Creating loss function...")
    class_weights = None
    if config.loss.use_class_weights:
        try:
            class_weights = compute_class_weights()
            print(f"Loaded class weights (min: {class_weights.min():.3f}, max: {class_weights.max():.3f})")
        except Exception as e:
            print(f"Could not load class weights: {e}")
    
    loss_fn = create_loss_function(config.loss, class_weights, device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Create scheduler
    scheduler = None
    if config.training.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.scheduler_params["T_max"],
            eta_min=config.training.scheduler_params.get("eta_min", 1e-6)
        )
    elif config.training.scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=config.training.scheduler_params["patience"],
            factor=config.training.scheduler_params["factor"]
        )
    
    # Create feature normalizer
    normalizer = FeatureNormalizer()
    # Note: In production, fit normalizer on training data
    
    # Create metrics tracker
    tracker = MetricsTracker()
    
    # Training loop
    print("\nStarting training...")
    best_val_score = -float('inf')
    patience_counter = 0
    
    for epoch in range(1, config.training.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.training.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        tracker.reset_epoch()
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=config.training.grad_clip,
            metadata_dataset=train_dataset.metadata_dataset,
            tracker=tracker,
            normalizer=normalizer,
            epoch=epoch,
            log_interval=config.log_interval
        )
        
        print(f"\nTrain metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Validate
        if epoch % config.validate_interval == 0:
            val_metrics = validate_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                metadata_dataset=val_dataset.metadata_dataset,
                normalizer=normalizer,
                threshold=config.inference.default_threshold
            )
            
            print(f"\nValidation metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Check for improvement
            val_score = val_metrics.get("room_score", val_metrics.get("f1", 0))
            is_best = val_score > best_val_score
            
            if is_best:
                best_val_score = val_score
                patience_counter = 0
                print(f"✓ New best validation score: {best_val_score:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{config.training.early_stopping_patience})")
            
            # Save checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={**train_metrics, **val_metrics},
                config=config,
                checkpoint_dir=config.checkpoint_dir,
                is_best=is_best
            )
            
            # Early stopping
            if patience_counter >= config.training.early_stopping_patience:
                print("\nEarly stopping triggered!")
                break
            
            # Update scheduler (after validation, before next epoch)
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_score)
                else:
                    scheduler.step()
        else:
            # If not validating this epoch, still step the scheduler
            if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
        
        # Reshuffle dataset
        if config.sampling.reshuffle_each_epoch:
            train_dataset.shuffle()
    
    print("\nTraining complete!")
    print(f"Best validation score: {best_val_score:.4f}")
    
    # Save final metrics
    tracker.save(log_dir / "training_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deep Sets Autoencoder")
    parser.add_argument("--config", type=str, default="default", 
                       choices=["default", "fast", "production"],
                       help="Configuration preset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == "default":
        from config.train_config import get_default_config
        config = get_default_config()
    elif args.config == "fast":
        from config.train_config import get_fast_experiment_config
        config = get_fast_experiment_config()
    elif args.config == "production":
        from config.train_config import get_production_config
        config = get_production_config()
    
    # Override with command-line arguments
    if args.device:
        config.device = args.device
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    # Run training
    main(config)
