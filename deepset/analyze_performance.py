"""
Comprehensive performance analysis script.

This script analyzes the trained Deep Sets model to understand:
1. Why F1=0.36 gives negative competition scores
2. The relationship between threshold and room score
3. Per-operation performance (which operations have high FP/FN rates)
4. Empty room detection performance
5. Optimal threshold for competition scoring
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import sys
from pathlib import Path
import importlib.util

# Load deepset modules directly
deepset_dir = Path(__file__).parent

# Load config
spec = importlib.util.spec_from_file_location("deepset_config", deepset_dir / "config.py")
deepset_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepset_config)
Config = deepset_config.Config

# Load model
spec = importlib.util.spec_from_file_location("deepset_model", deepset_dir / "model.py")
deepset_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepset_model)
DeepSetsModel = deepset_model.DeepSetsModel

# Load data_loader
spec = importlib.util.spec_from_file_location("deepset_data_loader", deepset_dir / "data_loader.py")
deepset_data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepset_data_loader)
get_dataloaders = deepset_data_loader.get_dataloaders

# Load utils
spec = importlib.util.spec_from_file_location("deepset_utils", deepset_dir / "utils.py")
deepset_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepset_utils)
compute_metrics = deepset_utils.compute_metrics
compute_room_score = deepset_utils.compute_room_score
AverageMeter = deepset_utils.AverageMeter


def analyze_threshold_vs_room_score(model, val_loader, thresholds, device='cuda'):
    """
    Analyze how different thresholds affect room score.
    
    This is CRITICAL because:
    - Training optimizes F1 (balanced precision/recall)
    - Competition uses room score with asymmetric penalties
    - FN penalty (0.5) is 2x FP penalty (0.25)
    - But negative scores indicate too many FPs
    
    Returns dict with metrics for each threshold.
    """
    model.eval()
    
    results = {}
    
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Analyzing threshold: {threshold:.2f}")
        print(f"{'='*60}")
        
        all_preds = []
        all_targets = []
        room_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Threshold {threshold:.2f}"):
                X = batch['X'].to(device)
                Y = batch['Y'].to(device)
                context = batch['context'].to(device)
                context_mask = batch['context_mask'].to(device)
                
                # Get predictions
                logits = model(X, context, context_mask)
                probs = torch.sigmoid(logits)
                
                # Apply threshold
                preds = (probs >= threshold).float()
                
                # Compute room scores for each sample in batch
                for i in range(len(preds)):
                    pred_ops = preds[i].cpu().numpy()
                    target_ops = Y[i].cpu().numpy()
                    
                    # Convert binary arrays to lists of operation indices
                    pred_indices = np.where(pred_ops > 0)[0].tolist()
                    target_indices = np.where(target_ops > 0)[0].tolist()
                    
                    # Compute room score for this single room
                    room_score = compute_room_score([pred_indices], [target_indices])
                    room_scores.append(room_score)
                    
                    all_preds.append(pred_ops)
                    all_targets.append(target_ops)
        
        # Compute aggregate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        metrics = compute_metrics(
            torch.tensor(all_preds),
            torch.tensor(all_targets),
            threshold=threshold
        )
        
        # Add room score statistics
        room_scores = np.array(room_scores)
        metrics['mean_room_score'] = room_scores.mean()
        metrics['median_room_score'] = np.median(room_scores)
        metrics['std_room_score'] = room_scores.std()
        metrics['min_room_score'] = room_scores.min()
        metrics['max_room_score'] = room_scores.max()
        metrics['negative_score_pct'] = (room_scores < 0).mean() * 100
        
        results[threshold] = metrics
        
        # Print summary
        print(f"\nMetrics at threshold {threshold:.2f}:")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1:         {metrics['f1']:.4f}")
        print(f"  TP:         {metrics['tp']:.0f}")
        print(f"  FP:         {metrics['fp']:.0f}")
        print(f"  FN:         {metrics['fn']:.0f}")
        print(f"  TN:         {metrics['tn']:.0f}")
        print(f"\nRoom Score Statistics:")
        print(f"  Mean:       {metrics['mean_room_score']:.4f}")
        print(f"  Median:     {metrics['median_room_score']:.4f}")
        print(f"  Std:        {metrics['std_room_score']:.4f}")
        print(f"  Min:        {metrics['min_room_score']:.4f}")
        print(f"  Max:        {metrics['max_room_score']:.4f}")
        print(f"  % Negative: {metrics['negative_score_pct']:.2f}%")
        
        # Calculate expected room score from TP/FP/FN
        expected_score = (metrics['tp'] * 1.0 - 
                         metrics['fp'] * 0.25 - 
                         metrics['fn'] * 0.5)
        num_samples = len(room_scores)
        expected_mean = expected_score / num_samples
        print(f"\nExpected mean room score from TP/FP/FN: {expected_mean:.4f}")
        print(f"Actual mean room score: {metrics['mean_room_score']:.4f}")
        print(f"Difference: {metrics['mean_room_score'] - expected_mean:.4f}")
    
    return results


def analyze_per_operation_performance(model, val_loader, threshold, device='cuda', 
                                      operation_names=None):
    """
    Analyze performance for each operation individually.
    
    This helps identify:
    - Which operations have high FP rates (predicted too often)
    - Which operations have high FN rates (missed too often)
    - Rare operations that might need special handling
    """
    model.eval()
    
    num_operations = 388  # From config
    
    # Track per-operation stats
    op_tp = np.zeros(num_operations)
    op_fp = np.zeros(num_operations)
    op_fn = np.zeros(num_operations)
    op_tn = np.zeros(num_operations)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Per-operation analysis"):
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            context = batch['context'].to(device)
            context_mask = batch['context_mask'].to(device)
            
            logits = model(X, context, context_mask)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            
            # Compute confusion matrix for each operation
            for i in range(num_operations):
                pred_op = preds[:, i].cpu().numpy()
                target_op = Y[:, i].cpu().numpy()
                
                op_tp[i] += np.sum((pred_op == 1) & (target_op == 1))
                op_fp[i] += np.sum((pred_op == 1) & (target_op == 0))
                op_fn[i] += np.sum((pred_op == 0) & (target_op == 1))
                op_tn[i] += np.sum((pred_op == 0) & (target_op == 0))
    
    # Compute per-operation metrics
    results = []
    for i in range(num_operations):
        precision = op_tp[i] / (op_tp[i] + op_fp[i]) if (op_tp[i] + op_fp[i]) > 0 else 0
        recall = op_tp[i] / (op_tp[i] + op_fn[i]) if (op_tp[i] + op_fn[i]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Compute room score contribution
        room_score_contrib = op_tp[i] * 1.0 - op_fp[i] * 0.25 - op_fn[i] * 0.5
        
        # Frequency in targets
        freq = (op_tp[i] + op_fn[i]) / len(val_loader.dataset)
        
        results.append({
            'operation_id': i,
            'operation_name': operation_names[i] if operation_names else f"Op_{i}",
            'tp': int(op_tp[i]),
            'fp': int(op_fp[i]),
            'fn': int(op_fn[i]),
            'tn': int(op_tn[i]),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'frequency': freq,
            'room_score_contrib': room_score_contrib
        })
    
    return results


def analyze_empty_rooms(model, val_loader, threshold, device='cuda'):
    """
    Analyze performance on empty rooms (no operations).
    
    Empty rooms are critical because:
    - Correctly predicting empty room = +1 point
    - Any FP on empty room = -0.25 per operation (pure penalty)
    - Model might struggle to predict "nothing"
    """
    model.eval()
    
    empty_room_stats = {
        'total_empty': 0,
        'correctly_predicted_empty': 0,
        'false_positives_on_empty': [],
        'num_fps_per_empty_room': []
    }
    
    non_empty_stats = {
        'total_non_empty': 0,
        'room_scores': []
    }
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Empty room analysis"):
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            context = batch['context'].to(device)
            context_mask = batch['context_mask'].to(device)
            
            logits = model(X, context, context_mask)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            
            for i in range(len(preds)):
                pred_ops = preds[i].cpu().numpy()
                target_ops = Y[i].cpu().numpy()
                
                # Convert to operation indices
                pred_indices = np.where(pred_ops > 0)[0].tolist()
                target_indices = np.where(target_ops > 0)[0].tolist()
                
                is_empty = len(target_indices) == 0
                
                if is_empty:
                    empty_room_stats['total_empty'] += 1
                    num_pred_ops = len(pred_indices)
                    
                    if num_pred_ops == 0:
                        empty_room_stats['correctly_predicted_empty'] += 1
                    else:
                        empty_room_stats['false_positives_on_empty'].append(int(num_pred_ops))
                    
                    empty_room_stats['num_fps_per_empty_room'].append(int(num_pred_ops))
                else:
                    non_empty_stats['total_non_empty'] += 1
                    room_score = compute_room_score([pred_indices], [target_indices])
                    non_empty_stats['room_scores'].append(room_score)
    
    # Compute statistics
    total_empty = empty_room_stats['total_empty']
    correct_empty = empty_room_stats['correctly_predicted_empty']
    
    if total_empty > 0:
        empty_accuracy = correct_empty / total_empty
        avg_fps_on_empty = np.mean(empty_room_stats['num_fps_per_empty_room'])
    else:
        empty_accuracy = 0
        avg_fps_on_empty = 0
    
    print(f"\n{'='*60}")
    print(f"Empty Room Analysis")
    print(f"{'='*60}")
    print(f"Total empty rooms: {total_empty}")
    print(f"Correctly predicted empty: {correct_empty} ({empty_accuracy*100:.2f}%)")
    print(f"Average FPs on empty rooms: {avg_fps_on_empty:.2f}")
    print(f"\nNon-empty rooms: {non_empty_stats['total_non_empty']}")
    if non_empty_stats['room_scores']:
        print(f"Mean room score (non-empty): {np.mean(non_empty_stats['room_scores']):.4f}")
    
    return empty_room_stats, non_empty_stats


def plot_threshold_analysis(results, output_path='deepset/threshold_analysis.png'):
    """Plot how metrics change with threshold."""
    thresholds = sorted(results.keys())
    
    metrics_to_plot = ['precision', 'recall', 'f1', 'mean_room_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [results[t][metric] for t in thresholds]
        axes[idx].plot(thresholds, values, 'o-', linewidth=2, markersize=8)
        axes[idx].set_xlabel('Threshold', fontsize=12)
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_title(f'{metric.replace("_", " ").title()} vs Threshold', fontsize=14)
        
        # Mark optimal
        optimal_idx = np.argmax(values)
        optimal_t = thresholds[optimal_idx]
        optimal_v = values[optimal_idx]
        axes[idx].axvline(optimal_t, color='r', linestyle='--', alpha=0.5)
        axes[idx].text(optimal_t, optimal_v, f'  Max: {optimal_v:.3f}\n  @ {optimal_t:.2f}',
                      fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved threshold analysis plot to: {output_path}")
    plt.close()


def main():
    """Run comprehensive performance analysis."""
    print("="*80)
    print("DEEP SETS MODEL - COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Load config and model
    config = deepset_config.get_config('debug')  # Use same as training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"Config preset: debug")
    
    # Load model
    model_path = Path("checkpoints/best_model.pt")  # Root checkpoints folder
    if not model_path.exists():
        print(f"\nERROR: Model not found at {model_path}")
        print("Please train the model first using: python deepset/train.py")
        return
    
    model = DeepSetsModel(
        num_operations=config.model.num_operations,
        num_rooms=config.model.num_rooms,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        pooling=config.model.pooling,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded model from: {model_path}")
    print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'best_f1' in checkpoint:
        print(f"Best validation F1: {checkpoint['best_f1']:.4f}")
    elif 'val_f1' in checkpoint:
        print(f"Validation F1: {checkpoint['val_f1']:.4f}")
    else:
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Load validation data
    print("\nLoading validation data...")
    dataloaders = get_dataloaders(
        config.data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    val_loader = dataloaders['val']
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # 1. Threshold analysis
    print("\n" + "="*80)
    print("TASK 1: THRESHOLD VS ROOM SCORE ANALYSIS")
    print("="*80)
    
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = analyze_threshold_vs_room_score(model, val_loader, thresholds, device)
    
    # Find optimal threshold for room score
    optimal_threshold = max(results.keys(), key=lambda t: results[t]['mean_room_score'])
    print(f"\n{'='*80}")
    print(f"OPTIMAL THRESHOLD FOR ROOM SCORE: {optimal_threshold:.2f}")
    print(f"Mean room score: {results[optimal_threshold]['mean_room_score']:.4f}")
    print(f"F1 score: {results[optimal_threshold]['f1']:.4f}")
    print(f"{'='*80}")
    
    # Plot results
    plot_threshold_analysis(results)
    
    # 2. Per-operation analysis
    print("\n" + "="*80)
    print("TASK 2: PER-OPERATION PERFORMANCE ANALYSIS")
    print("="*80)
    
    op_results = analyze_per_operation_performance(
        model, val_loader, optimal_threshold, device
    )
    
    # Sort by room score contribution (most negative first)
    op_results_sorted = sorted(op_results, key=lambda x: x['room_score_contrib'])
    
    print("\nTop 10 operations with MOST NEGATIVE room score contribution:")
    print(f"{'Rank':<6} {'OpID':<6} {'TP':<6} {'FP':<6} {'FN':<6} {'Score':<10} {'Freq':<8}")
    print("-" * 60)
    for rank, op in enumerate(op_results_sorted[:10], 1):
        print(f"{rank:<6} {op['operation_id']:<6} {op['tp']:<6} {op['fp']:<6} "
              f"{op['fn']:<6} {op['room_score_contrib']:<10.2f} {op['frequency']:<8.4f}")
    
    print("\nTop 10 operations with MOST POSITIVE room score contribution:")
    print(f"{'Rank':<6} {'OpID':<6} {'TP':<6} {'FP':<6} {'FN':<6} {'Score':<10} {'Freq':<8}")
    print("-" * 60)
    for rank, op in enumerate(reversed(op_results_sorted[-10:]), 1):
        print(f"{rank:<6} {op['operation_id']:<6} {op['tp']:<6} {op['fp']:<6} "
              f"{op['fn']:<6} {op['room_score_contrib']:<10.2f} {op['frequency']:<8.4f}")
    
    # 3. Empty room analysis
    print("\n" + "="*80)
    print("TASK 3: EMPTY ROOM DETECTION ANALYSIS")
    print("="*80)
    
    empty_stats, non_empty_stats = analyze_empty_rooms(
        model, val_loader, optimal_threshold, device
    )
    
    # Save detailed results
    output_file = Path("deepset/performance_analysis.txt")
    with open(output_file, 'w') as f:
        f.write("DEEP SETS MODEL - PERFORMANCE ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("THRESHOLD ANALYSIS\n")
        f.write("-" * 80 + "\n")
        for threshold in sorted(results.keys()):
            metrics = results[threshold]
            f.write(f"\nThreshold: {threshold:.2f}\n")
            f.write(f"  F1:              {metrics['f1']:.4f}\n")
            f.write(f"  Precision:       {metrics['precision']:.4f}\n")
            f.write(f"  Recall:          {metrics['recall']:.4f}\n")
            f.write(f"  Mean room score: {metrics['mean_room_score']:.4f}\n")
            f.write(f"  % Negative:      {metrics['negative_score_pct']:.2f}%\n")
        
        f.write(f"\n\nOPTIMAL THRESHOLD: {optimal_threshold:.2f}\n")
        f.write(f"Mean room score: {results[optimal_threshold]['mean_room_score']:.4f}\n")
        
    print(f"\nSaved detailed results to: {output_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print(f"1. Optimal threshold for room score: {optimal_threshold:.2f}")
    print(f"2. Mean room score at optimal threshold: {results[optimal_threshold]['mean_room_score']:.4f}")
    print(f"3. Empty room accuracy: {empty_stats['correctly_predicted_empty']}/{empty_stats['total_empty']}")
    print("\nNext steps:")
    print("- If mean room score is still negative, increase threshold further")
    print("- Focus on reducing FPs on empty rooms")
    print("- Consider per-operation thresholds for operations with high FP rates")
    print("- Retrain with custom loss that weights FN 2x more than FP")


if __name__ == '__main__':
    main()
