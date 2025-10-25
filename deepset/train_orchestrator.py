"""
Training orchestrator - Runs multiple training configurations in sequence.

This script trains models with different optimizations:
1. Optimized + Competition Focal Loss (moderate, 50 epochs)
2. Ultimate + All Features (aggressive, 60 epochs)
"""

import subprocess
import time
from pathlib import Path
import json

# Training configurations
TRAINING_CONFIGS = [
    {
        "name": "Optimized + Competition Focal",
        "config": "optimized",
        "loss": "competition_focal",
        "description": "Moderate model with competition-aligned focal loss",
        "expected_improvement": "+100-200 points",
    },
    {
        "name": "Ultimate + Auxiliary Task",
        "config": "ultimate",
        "loss": "competition_focal",
        "description": "Large model with empty room classifier + competition loss",
        "expected_improvement": "+200-350 points",
    },
    {
        "name": "Aggressive + Competition Aligned",
        "config": "aggressive",
        "loss": "competition_aligned",
        "description": "Very conservative model optimized for FP reduction",
        "expected_improvement": "+150-250 points",
    },
]


def run_training(config_name: str, loss_type: str):
    """Run training with specified configuration."""
    cmd = [
        "uv", "run", "python", "deepset/train.py",
        "--config", config_name,
        "--loss", loss_type,
    ]
    
    print(f"\n{'='*80}")
    print(f"Starting training: {config_name} with {loss_type}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Run training
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print(f"\n{'='*80}")
    print(f"Training completed in {hours}h {minutes}m")
    print(f"{'='*80}\n")
    
    return result.returncode == 0


def main():
    """Run all training configurations."""
    results = {}
    
    print("="*80)
    print("COMPREHENSIVE MODEL TRAINING - Competition Optimization")
    print("="*80)
    print(f"\nTotal configurations: {len(TRAINING_CONFIGS)}")
    print("\nScheduled trainings:")
    for i, cfg in enumerate(TRAINING_CONFIGS, 1):
        print(f"\n{i}. {cfg['name']}")
        print(f"   Config: {cfg['config']}, Loss: {cfg['loss']}")
        print(f"   Description: {cfg['description']}")
        print(f"   Expected: {cfg['expected_improvement']}")
    
    input("\nPress Enter to start training sequence...")
    
    for i, cfg in enumerate(TRAINING_CONFIGS, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Training {i}/{len(TRAINING_CONFIGS)}: {cfg['name']}")
        print(f"{'#'*80}\n")
        
        success = run_training(cfg['config'], cfg['loss'])
        results[cfg['name']] = {
            'success': success,
            'config': cfg['config'],
            'loss': cfg['loss'],
        }
        
        if not success:
            print(f"\n❌ Training failed for {cfg['name']}")
            print("Stopping training sequence.")
            break
        
        print(f"\n✅ Training completed for {cfg['name']}")
        
        # Brief pause between trainings
        if i < len(TRAINING_CONFIGS):
            print("\nWaiting 10 seconds before next training...")
            time.sleep(10)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}\n")
    
    for name, result in results.items():
        status = "✅ Success" if result['success'] else "❌ Failed"
        print(f"{status}: {name} ({result['config']} + {result['loss']})")
    
    # Save results
    results_file = Path("deepset/training_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("\nNext steps:")
    print("1. Check checkpoints/ directory for trained models")
    print("2. Run evaluate.py to generate submissions")
    print("3. Upload submissions to Kaggle for scoring")


if __name__ == "__main__":
    main()
