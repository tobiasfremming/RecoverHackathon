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
        
        # Check if model uses auxiliary task
        use_empty_room_head = config.loss.use_auxiliary_loss if hasattr(config.loss, 'use_auxiliary_loss') else False
        
        # Create model
        self.model = DeepSetsModel(
            num_operations=config.model.num_operations,
            num_rooms=config.model.num_rooms,
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            pooling=config.model.pooling,
            use_empty_room_head=use_empty_room_head,
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
        use_dual_threshold: bool = False,
        empty_room_threshold: float = 0.60,
        empty_room_max_prob: float = 0.35,
        suppress_problematic_ops: bool = False,
        problematic_ops: Optional[List[int]] = None,
        problematic_threshold: float = 0.55,
    ):
        """
        Generate submission file.
        
        Args:
            dataloader: Test dataloader
            output_path: Path to save submission
            threshold: Classification threshold (uses checkpoint threshold if None)
            use_dual_threshold: Use different thresholds for empty vs non-empty rooms
            empty_room_threshold: Higher threshold for suspected empty rooms
            empty_room_max_prob: If max(probs) < this, treat as empty room
            suppress_problematic_ops: Apply higher threshold to operations with high FP rate
            problematic_ops: List of operation IDs to suppress (uses default if None)
            problematic_threshold: Threshold for problematic operations
        """
        if threshold is None:
            threshold = self.threshold
        
        # Default problematic operations (high FP rate from analysis)
        if suppress_problematic_ops and problematic_ops is None:
            problematic_ops = [260, 108, 204, 257, 259, 154, 262, 258, 103, 112]
        
        if use_dual_threshold:
            print(f"\nGenerating submission with DUAL-THRESHOLD strategy:")
            print(f"  Empty rooms (max_prob < {empty_room_max_prob}): threshold={empty_room_threshold:.2f}")
            print(f"  Non-empty rooms: threshold={threshold:.2f}")
            if suppress_problematic_ops:
                print(f"  Problematic operations {problematic_ops}: threshold={problematic_threshold:.2f}")
        else:
            print(f"\nGenerating submission with threshold {threshold:.2f}...")
            if suppress_problematic_ops:
                print(f"  Problematic operations {problematic_ops}: threshold={problematic_threshold:.2f}")
        
        all_probs = []
        all_empty_room_probs = []
        has_empty_room_head = self.model.use_empty_room_head if hasattr(self.model, 'use_empty_room_head') else False
        
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move to device
            X = batch["X"].to(self.device)
            context = batch["context"].to(self.device)
            context_mask = batch["context_mask"].to(self.device)
            
            # Forward pass
            if has_empty_room_head:
                logits, empty_room_logits = self.model(
                    X, context, context_mask, return_empty_room_logits=True
                )
                empty_room_probs = torch.sigmoid(empty_room_logits)
                all_empty_room_probs.append(empty_room_probs.cpu())
            else:
                logits = self.model(X, context, context_mask)
            
            probs = torch.sigmoid(logits)
            
            # Store predictions
            all_probs.append(probs.cpu())
        
        # Concatenate all predictions
        all_probs = torch.cat(all_probs, dim=0)
        if has_empty_room_head:
            all_empty_room_probs = torch.cat(all_empty_room_probs, dim=0)
            print(f"  Using empty room classifier (mean prob: {all_empty_room_probs.mean():.3f})")
        
        # Apply dual-threshold strategy if enabled
        if use_dual_threshold or suppress_problematic_ops or has_empty_room_head:
            preds_list = []
            empty_room_count = 0
            suppressed_count = 0
            
            for idx, room_probs in enumerate(all_probs):
                max_prob = room_probs.max().item()
                
                # Use empty room classifier if available
                if has_empty_room_head:
                    empty_prob = all_empty_room_probs[idx].item()
                    is_likely_empty = empty_prob > 0.5
                else:
                    is_likely_empty = max_prob < empty_room_max_prob
                
                # Determine base threshold
                if use_dual_threshold and is_likely_empty:
                    room_threshold = empty_room_threshold
                    empty_room_count += 1
                else:
                    room_threshold = threshold
                
                # Get base predictions
                predictions_mask = room_probs >= room_threshold
                
                # Apply per-operation suppression if enabled
                if suppress_problematic_ops:
                    for op_id in problematic_ops:
                        if predictions_mask[op_id] and room_probs[op_id] < problematic_threshold:
                            predictions_mask[op_id] = False
                            suppressed_count += 1
                
                # Get predictions for this room
                predictions = predictions_mask.nonzero(as_tuple=True)[0].tolist()
                preds_list.append(predictions)
            
            if use_dual_threshold:
                print(f"\n  Detected {empty_room_count} suspected empty rooms ({100*empty_room_count/len(all_probs):.1f}%)")
            if suppress_problematic_ops:
                print(f"  Suppressed {suppressed_count} problematic operation predictions")
        else:
            # Convert to operation codes (standard method)
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
        "--use-dual-threshold",
        action="store_true",
        help="Use dual-threshold strategy (higher threshold for suspected empty rooms)",
    )
    parser.add_argument(
        "--empty-room-threshold",
        type=float,
        default=0.60,
        help="Threshold for suspected empty rooms (default: 0.60)",
    )
    parser.add_argument(
        "--empty-room-max-prob",
        type=float,
        default=0.35,
        help="Max probability to classify as empty room (default: 0.35)",
    )
    parser.add_argument(
        "--suppress-problematic-ops",
        action="store_true",
        help="Apply higher threshold to operations with high FP rate",
    )
    parser.add_argument(
        "--problematic-threshold",
        type=float,
        default=0.55,
        help="Threshold for problematic operations (default: 0.55)",
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
            use_dual_threshold=args.use_dual_threshold,
            empty_room_threshold=args.empty_room_threshold,
            empty_room_max_prob=args.empty_room_max_prob,
            suppress_problematic_ops=args.suppress_problematic_ops,
            problematic_threshold=args.problematic_threshold,
        )


if __name__ == "__main__":
    main()
