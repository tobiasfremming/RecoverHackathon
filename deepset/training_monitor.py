"""
Training monitoring and analysis script.

Tracks multiple metrics during training:
- Standard metrics (F1, precision, recall)
- Competition room score
- Empty room accuracy
- Per-operation FP/FN rates
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import json


class TrainingMonitor:
    """Monitor training progress with competition-specific metrics."""
    
    def __init__(self, checkpoint_dir: str = "deepset/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
            "val_room_score": [],
            "val_empty_room_acc": [],
            "learning_rate": [],
        }
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Add metrics for current epoch."""
        self.history["epoch"].append(epoch)
        
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(self.history["epoch"], self.history["train_loss"], label="Train Loss")
        if self.history["val_loss"]:
            ax.plot(self.history["epoch"], self.history["val_loss"], label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 score
        ax = axes[0, 1]
        ax.plot(self.history["epoch"], self.history["val_f1"], label="Val F1", color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.set_title("Validation F1 Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Room score
        ax = axes[1, 0]
        ax.plot(self.history["epoch"], self.history["val_room_score"], label="Val Room Score", color="purple")
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label="Zero line")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Room Score")
        ax.set_title("Validation Room Score (Competition Metric)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Empty room accuracy
        ax = axes[1, 1]
        if self.history["val_empty_room_acc"]:
            ax.plot(self.history["epoch"], self.history["val_empty_room_acc"], label="Empty Room Acc", color="orange")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Empty Room Detection Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")
        plt.close()
    
    def save_history(self, save_path: str = "training_history.json"):
        """Save training history to JSON."""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {save_path}")
    
    def print_summary(self):
        """Print training summary."""
        if not self.history["epoch"]:
            print("No training history yet.")
            return
        
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        
        # Find best epoch
        best_f1_idx = np.argmax(self.history["val_f1"])
        best_room_score_idx = np.argmax(self.history["val_room_score"]) if self.history["val_room_score"] else 0
        
        print(f"\nBest F1 Score:")
        print(f"  Epoch: {self.history['epoch'][best_f1_idx]}")
        print(f"  F1: {self.history['val_f1'][best_f1_idx]:.4f}")
        print(f"  Precision: {self.history['val_precision'][best_f1_idx]:.4f}")
        print(f"  Recall: {self.history['val_recall'][best_f1_idx]:.4f}")
        
        if self.history["val_room_score"]:
            print(f"\nBest Room Score:")
            print(f"  Epoch: {self.history['epoch'][best_room_score_idx]}")
            print(f"  Room Score: {self.history['val_room_score'][best_room_score_idx]:.4f}")
            print(f"  F1 at this epoch: {self.history['val_f1'][best_room_score_idx]:.4f}")
        
        # Final metrics
        print(f"\nFinal Metrics (Epoch {self.history['epoch'][-1]}):")
        print(f"  Train Loss: {self.history['train_loss'][-1]:.4f}")
        if self.history["val_loss"]:
            print(f"  Val Loss: {self.history['val_loss'][-1]:.4f}")
        print(f"  Val F1: {self.history['val_f1'][-1]:.4f}")
        if self.history["val_room_score"]:
            print(f"  Val Room Score: {self.history['val_room_score'][-1]:.4f}")
        
        print("=" * 70)


def compare_models(checkpoint_paths: List[str], labels: List[str]):
    """Compare multiple trained models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    for path, label in zip(checkpoint_paths, labels):
        if not Path(path).exists():
            print(f"\n{label}: Checkpoint not found at {path}")
            continue
        
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        
        print(f"\n{label}:")
        print(f"  Path: {path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Val F1: {checkpoint.get('best_val_f1', 'N/A'):.4f}")
        print(f"  Best Threshold: {checkpoint.get('best_threshold', 'N/A'):.2f}")
        
        # Model size
        if 'model_state_dict' in checkpoint:
            num_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
            print(f"  Parameters: {num_params:,}")
        
        # Config
        if 'config' in checkpoint:
            config = checkpoint['config']
            if 'model' in config:
                print(f"  Embedding dim: {config['model'].get('embedding_dim', 'N/A')}")
                print(f"  Hidden dim: {config['model'].get('hidden_dim', 'N/A')}")
            if 'loss' in config:
                print(f"  Focal alpha: {config['loss'].get('focal_alpha', 'N/A')}")
                print(f"  Focal gamma: {config['loss'].get('focal_gamma', 'N/A')}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Test monitor
    monitor = TrainingMonitor()
    
    # Simulate some training
    for epoch in range(1, 11):
        metrics = {
            "train_loss": 0.5 - epoch * 0.03,
            "val_loss": 0.55 - epoch * 0.025,
            "val_f1": 0.2 + epoch * 0.02,
            "val_precision": 0.25 + epoch * 0.015,
            "val_recall": 0.18 + epoch * 0.025,
            "val_room_score": -0.1 + epoch * 0.035,
            "val_empty_room_acc": 0.6 + epoch * 0.02,
            "learning_rate": 1e-4,
        }
        monitor.update(epoch, metrics)
    
    monitor.print_summary()
    monitor.plot_training_curves("test_training_curves.png")
    monitor.save_history("test_history.json")
    
    print("\n✓ Monitor test completed!")
