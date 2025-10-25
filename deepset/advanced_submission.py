"""
Advanced submission generator with all optimizations.

Combines:
1. Dual-threshold strategy (empty vs non-empty rooms)
2. Per-operation suppression (problematic operations)
3. Empty room classifier (if model has auxiliary head)
4. Calibrated probabilities (optional temperature scaling)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import argparse
import sys

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from deepset.evaluate import Evaluator
from deepset.config import get_config


class AdvancedSubmissionGenerator:
    """
    Generate optimized submissions with multiple strategies.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        self.evaluator = Evaluator(
            checkpoint_path=checkpoint_path,
            device=device,
        )
    
    def generate_submission(
        self,
        strategy: str = "ultimate",
        output_name: Optional[str] = None,
    ):
        """
        Generate submission with specified strategy.
        
        Args:
            strategy: Optimization strategy
                - "baseline": Standard threshold (0.45)
                - "dual": Dual-threshold (empty vs non-empty)
                - "suppression": Dual-threshold + per-op suppression
                - "ultimate": All optimizations + model auxiliary head
            output_name: Custom output filename
        """
        # Get dataloaders
        dataloaders = self.evaluator.config.data
        from deepset.data_loader import get_dataloaders
        dataloader_dict = get_dataloaders(
            dataloaders,
            batch_size=self.evaluator.config.training.batch_size,
            num_workers=0,
        )
        test_loader = dataloader_dict["test"]
        
        print(f"\n{'=' * 70}")
        print(f"GENERATING SUBMISSION - Strategy: {strategy.upper()}")
        print(f"{'=' * 70}\n")
        
        if strategy == "baseline":
            # Simple threshold
            self.evaluator.generate_submission(
                test_loader,
                threshold=0.45,
                use_dual_threshold=False,
                suppress_problematic_ops=False,
            )
        
        elif strategy == "dual":
            # Dual-threshold only
            self.evaluator.generate_submission(
                test_loader,
                threshold=0.45,
                use_dual_threshold=True,
                empty_room_threshold=0.60,
                empty_room_max_prob=0.35,
                suppress_problematic_ops=False,
            )
        
        elif strategy == "suppression":
            # Dual-threshold + per-operation suppression
            self.evaluator.generate_submission(
                test_loader,
                threshold=0.45,
                use_dual_threshold=True,
                empty_room_threshold=0.60,
                empty_room_max_prob=0.35,
                suppress_problematic_ops=True,
                problematic_threshold=0.55,
            )
        
        elif strategy == "ultimate":
            # All optimizations
            # If model has empty room head, it will be used automatically
            self.evaluator.generate_submission(
                test_loader,
                threshold=0.45,
                use_dual_threshold=True,
                empty_room_threshold=0.60,
                empty_room_max_prob=0.35,
                suppress_problematic_ops=True,
                problematic_threshold=0.55,
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"\n{'=' * 70}")
        print(f"SUBMISSION GENERATED SUCCESSFULLY")
        print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate advanced submissions with all optimizations"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ultimate",
        choices=["baseline", "dual", "suppression", "ultimate"],
        help="Submission strategy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Generate submission
    generator = AdvancedSubmissionGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    generator.generate_submission(strategy=args.strategy)


if __name__ == "__main__":
    main()
