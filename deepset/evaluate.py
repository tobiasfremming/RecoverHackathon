"""
Evaluation and submission generation for Deep Sets model.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import torch
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from deepset.config import get_config, Config
from deepset.model import DeepSetsModel
from deepset.data_loader import get_dataloaders
from deepset.utils import (
    compute_metrics,
    compute_room_score,
    predictions_to_codes,
    find_optimal_threshold,
)


class Evaluator:
    """Evaluation and submission generator."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Config] = None,
        device: str = "cuda",
    ):
        self.device = device
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load config from checkpoint or use provided
        if config is None:
            config = Config(**checkpoint["config"])
        self.config = config
        
        # Create model
        self.model = DeepSetsModel(
            num_operations=config.model.num_operations,
            num_rooms=config.model.num_rooms,
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            pooling=config.model.pooling,
        ).to(device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Get threshold from checkpoint
        self.threshold = checkpoint.get("best_threshold", 0.5)
        
        print(f"Model loaded successfully")
        print(f"  Best threshold: {self.threshold:.2f}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        split_name: str = "val",
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader to evaluate on
            split_name: Name of split for printing
        
        Returns:
            metrics: Dictionary of metric values
        """
        print(f"\nEvaluating on {split_name} set...")
        
        all_probs = []
        all_targets = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            X = batch["X"].to(self.device)
            Y = batch["Y"].to(self.device)
            context = batch["context"].to(self.device)
            context_mask = batch["context_mask"].to(self.device)
            
            # Forward pass
            logits = self.model(X, context, context_mask)
            probs = torch.sigmoid(logits)
            
            # Store predictions
            all_probs.append(probs.cpu())
            all_targets.append(Y.cpu())
        
        # Concatenate all predictions
        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Find optimal threshold on this split
        best_threshold, best_f1 = find_optimal_threshold(
            all_probs, all_targets, metric="f1"
        )
        
        # Compute metrics with both thresholds
        metrics_checkpoint = compute_metrics(
            all_probs, all_targets, threshold=self.threshold
        )
        metrics_optimal = compute_metrics(
            all_probs, all_targets, threshold=best_threshold
        )
        
        # Print results
        print(f"\nResults on {split_name} set:")
        print(f"  With checkpoint threshold ({self.threshold:.2f}):")
        print(f"    F1: {metrics_checkpoint['f1']:.4f}")
        print(f"    Precision: {metrics_checkpoint['precision']:.4f}")
        print(f"    Recall: {metrics_checkpoint['recall']:.4f}")
        print(f"    TP: {metrics_checkpoint['tp']:.0f}, FP: {metrics_checkpoint['fp']:.0f}, FN: {metrics_checkpoint['fn']:.0f}")
        
        print(f"  With optimal threshold ({best_threshold:.2f}):")
        print(f"    F1: {metrics_optimal['f1']:.4f}")
        print(f"    Precision: {metrics_optimal['precision']:.4f}")
        print(f"    Recall: {metrics_optimal['recall']:.4f}")
        print(f"    TP: {metrics_optimal['tp']:.0f}, FP: {metrics_optimal['fp']:.0f}, FN: {metrics_optimal['fn']:.0f}")
        
        # Compute room score
        preds_list = predictions_to_codes(all_probs, threshold=self.threshold)
        targets_list = predictions_to_codes(all_targets, threshold=0.5)  # Ground truth
        
        room_score = compute_room_score(preds_list, targets_list)
        avg_room_score = room_score / len(preds_list)
        
        print(f"  Room Score: {room_score:.2f} (avg: {avg_room_score:.4f})")
        
        return {
            "f1": metrics_checkpoint["f1"],
            "precision": metrics_checkpoint["precision"],
            "recall": metrics_checkpoint["recall"],
            "room_score": room_score,
            "avg_room_score": avg_room_score,
            "optimal_threshold": best_threshold,
            "optimal_f1": metrics_optimal["f1"],
        }
    
    @torch.no_grad()
    def generate_submission(
        self,
        dataloader,
        output_path: str = "submission.csv",
        threshold: Optional[float] = None,
    ):
        """
        Generate submission file.
        
        Args:
            dataloader: Test dataloader
            output_path: Path to save submission
            threshold: Classification threshold (uses checkpoint threshold if None)
        """
        if threshold is None:
            threshold = self.threshold
        
        print(f"\nGenerating submission with threshold {threshold:.2f}...")
        
        all_probs = []
        
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move to device
            X = batch["X"].to(self.device)
            context = batch["context"].to(self.device)
            context_mask = batch["context_mask"].to(self.device)
            
            # Forward pass
            logits = self.model(X, context, context_mask)
            probs = torch.sigmoid(logits)
            
            # Store predictions
            all_probs.append(probs.cpu())
        
        # Concatenate all predictions
        all_probs = torch.cat(all_probs, dim=0)
        
        # Convert to operation codes
        preds_list = predictions_to_codes(all_probs, threshold=threshold)
        
        # Create predictions dictionary for HackathonDataset.create_submission()
        # Format: {room_id: [list of operation codes]}
        predictions = {idx: codes for idx, codes in enumerate(preds_list)}
        
        # Use HackathonDataset's create_submission method
        import sys
        from pathlib import Path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        import dataset.hackathon as hackathon_module
        
        # Get the test dataset
        test_dataset = hackathon_module.HackathonDataset(
            split="test",
            root="data",
        )
        
        # Create submission using dataset method (saves to submissions/ folder)
        test_dataset.create_submission(predictions)
        
        print(f"\n✓ Submission created successfully!")
        print(f"  Total rooms: {len(predictions)}")
        print(f"  Empty predictions: {sum(1 for codes in preds_list if len(codes) == 0)}")
        print(f"  Non-empty predictions: {sum(1 for codes in preds_list if len(codes) > 0)}")
        
        # Statistics
        num_ops_per_room = [len(codes) for codes in preds_list]
        print(f"  Avg operations per room: {sum(num_ops_per_room) / len(num_ops_per_room):.2f}")
        print(f"  Max operations per room: {max(num_ops_per_room)}")
        print(f"  Min operations per room: {min(num_ops_per_room)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Deep Sets model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration preset (uses checkpoint config if None)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (uses checkpoint threshold if None)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Split to evaluate on",
    )
    parser.add_argument(
        "--generate-submission",
        action="store_true",
        help="Generate submission file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output path for submission",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        config = get_config(args.config)
    
    # Create evaluator
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
    )
    
    # Load data
    if config is None:
        config = evaluator.config
    
    dataloaders = get_dataloaders(
        config.data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    
    # Evaluate
    if args.split in ["train", "val"]:
        evaluator.evaluate(dataloaders[args.split], split_name=args.split)
    
    # Generate submission
    if args.generate_submission:
        evaluator.generate_submission(
            dataloaders["test"],
            output_path=args.output,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
