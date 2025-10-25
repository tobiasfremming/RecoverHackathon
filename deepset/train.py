"""
Training script for Deep Sets model.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from deepset.config import get_config, Config
from deepset.model import DeepSetsModel
from deepset.data_loader import get_dataloaders
from deepset.utils import (
    FocalLoss,
    WeightedBCELoss,
    compute_class_weights,
    compute_metrics,
    AverageMeter,
    set_seed,
    find_optimal_threshold,
)


class Trainer:
    """Training manager for Deep Sets model."""
    
    def __init__(
        self,
        config: Config,
        device: str = "cuda",
        resume_from: Optional[str] = None,
    ):
        self.config = config
        self.device = device
        
        # Set seed
        set_seed(config.data.seed)
        
        # Create model
        self.model = DeepSetsModel(
            num_operations=config.model.num_operations,
            num_rooms=config.model.num_rooms,
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            pooling=config.model.pooling,
        ).to(device)
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Create dataloaders
        self.dataloaders = get_dataloaders(
            config.data,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
        )
        
        # Create loss function
        self.loss_fn = self._create_loss_fn()
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        
        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=config.training.scheduler_patience,
        )
        
        # Training state
        self.start_epoch = 0
        self.best_val_f1 = 0.0
        self.best_threshold = 0.5
        
        # Resume from checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def _create_loss_fn(self) -> nn.Module:
        """Create loss function based on config."""
        if self.config.loss.loss_type == "focal":
            return FocalLoss(
                alpha=self.config.loss.focal_alpha,
                gamma=self.config.loss.focal_gamma,
            )
        elif self.config.loss.loss_type == "bce":
            if self.config.loss.use_class_weights:
                # Compute weights from training data
                pos_weights = compute_class_weights(
                    self.dataloaders["train"].dataset,
                    scale=self.config.loss.pos_weight_scale,
                    device=self.device,
                )
                return WeightedBCELoss(pos_weights)
            else:
                return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss.loss_type}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        loss_meter = AverageMeter()
        
        pbar = tqdm(
            self.dataloaders["train"],
            desc=f"Epoch {epoch}/{self.config.training.num_epochs}",
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            X = batch["X"].to(self.device)
            Y = batch["Y"].to(self.device)
            context = batch["context"].to(self.device)
            context_mask = batch["context_mask"].to(self.device)
            
            # Forward pass
            logits = self.model(X, context, context_mask)
            loss = self.loss_fn(logits, Y)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN loss at batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )
            
            self.optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), X.size(0))
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
        
        return {"train_loss": loss_meter.avg}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        loss_meter = AverageMeter()
        all_probs = []
        all_targets = []
        
        for batch in tqdm(self.dataloaders["val"], desc="Validating"):
            # Move to device
            X = batch["X"].to(self.device)
            Y = batch["Y"].to(self.device)
            context = batch["context"].to(self.device)
            context_mask = batch["context_mask"].to(self.device)
            
            # Forward pass
            logits = self.model(X, context, context_mask)
            loss = self.loss_fn(logits, Y)
            
            # Update metrics
            loss_meter.update(loss.item(), X.size(0))
            
            # Store predictions
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_targets.append(Y.cpu())
        
        # Concatenate all predictions
        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Find optimal threshold
        best_threshold, best_f1 = find_optimal_threshold(
            all_probs, all_targets, metric="f1"
        )
        
        # Compute metrics with best threshold
        metrics = compute_metrics(all_probs, all_targets, threshold=best_threshold)
        
        return {
            "val_loss": loss_meter.avg,
            "val_f1": metrics["f1"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_accuracy": metrics["accuracy"],
            "best_threshold": best_threshold,
        }
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")
        
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch + 1)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_metrics["val_f1"])
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val F1: {val_metrics['val_f1']:.4f}")
            print(f"  Val Precision: {val_metrics['val_precision']:.4f}")
            print(f"  Val Recall: {val_metrics['val_recall']:.4f}")
            print(f"  Best Threshold: {val_metrics['best_threshold']:.2f}")
            
            # Save checkpoint
            is_best = val_metrics["val_f1"] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics["val_f1"]
                self.best_threshold = val_metrics["best_threshold"]
                self.save_checkpoint(epoch + 1, is_best=True)
                print(f"  ✓ New best model saved! (F1: {self.best_val_f1:.4f})")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
        
        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best Val F1: {self.best_val_f1:.4f}")
        print(f"Best Threshold: {self.best_threshold:.2f}")
        print("=" * 80 + "\n")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_f1": self.best_val_f1,
            "best_threshold": self.best_threshold,
            "config": self.config.__dict__,
        }
        
        if is_best:
            path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)
            print(f"  Saved best checkpoint to {path}")
        else:
            path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)
            print(f"  Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.start_epoch = checkpoint["epoch"]
        self.best_val_f1 = checkpoint["best_val_f1"]
        self.best_threshold = checkpoint["best_threshold"]
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {self.start_epoch}")
        print(f"  Best Val F1: {self.best_val_f1:.4f}")
        print(f"  Best Threshold: {self.best_threshold:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train Deep Sets model")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "fast", "strong", "debug"],
        help="Configuration preset",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    
    print("Configuration:")
    print(f"  Preset: {args.config}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.lr}")
    print(f"  Loss: {config.loss.loss_type}")
    print()
    
    # Create trainer
    trainer = Trainer(
        config=config,
        device=args.device,
        resume_from=args.resume,
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
