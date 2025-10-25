# Deep Sets Implementation - Usage Guide

## Overview

This is a complete, clean implementation of Deep Sets for the Recover Hackathon. The implementation is modular, well-documented, and designed to handle the competition's unique challenges:

- Multi-label binary classification (388 operations)
- Extreme class imbalance (0.74% positive rate)
- Permutation-invariant set processing
- Context from other rooms in projects

## Quick Start

### 1. Test the Pipeline

First, verify everything works:

```bash
python -m deepset.test_pipeline
```

This will test:
- Model architecture
- Data loading
- Loss functions
- Training step

### 2. Train the Model

#### Debug Mode (fast testing)
```bash
python -m deepset.train --config debug
```
- 2 epochs
- Small batch size
- Quick validation

#### Default Mode (recommended)
```bash
python -m deepset.train --config default
```
- 50 epochs
- Balanced configuration
- Good for initial experiments

#### Fast Mode (quick experiments)
```bash
python -m deepset.train --config fast
```
- 10 epochs
- Good for hyperparameter search

#### Strong Mode (best performance)
```bash
python -m deepset.train --config strong
```
- 100 epochs
- Larger model (256/512 dims)
- Higher capacity

### 3. Resume Training

```bash
python -m deepset.train --config default --resume checkpoints/checkpoint_epoch_25.pt
```

### 4. Evaluate Model

#### On validation set
```bash
python -m deepset.evaluate --checkpoint checkpoints/best_model.pt --split val
```

#### On test set
```bash
python -m deepset.evaluate --checkpoint checkpoints/best_model.pt --split test
```

### 5. Generate Submission

```bash
python -m deepset.evaluate --checkpoint checkpoints/best_model.pt --generate-submission --output submission.csv
```

Or with custom threshold:

```bash
python -m deepset.evaluate --checkpoint checkpoints/best_model.pt --generate-submission --threshold 0.3 --output submission_0.3.csv
```

## Project Structure

```
deepset/
├── __init__.py          # Package initialization
├── README.md            # Architecture documentation
├── USAGE.md             # This file
├── config.py            # Configuration system
├── model.py             # Deep Sets model
├── dataset.py           # Data loading
├── utils.py             # Loss functions, metrics
├── train.py             # Training script
├── evaluate.py          # Evaluation script
└── test_pipeline.py     # Pipeline tests
```

## Configuration Presets

### Default
```python
{
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 3,
    "pooling": "mean",
    "lr": 1e-4,
    "batch_size": 32,
    "num_epochs": 50,
    "loss_type": "focal",
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
}
```

### Fast (10 epochs)
Same as default but faster for experiments.

### Strong (higher capacity)
```python
{
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 4,
    "num_epochs": 100,
}
```

### Debug (2 epochs)
Quick testing configuration.

## Advanced Usage

### Custom Configuration

Modify `deepset/config.py` to create your own preset:

```python
def get_config(name: str = "default") -> Config:
    # ... existing code ...
    elif name == "my_config":
        return Config(
            model=ModelConfig(
                embedding_dim=192,
                hidden_dim=384,
                num_layers=4,
                pooling="max",
            ),
            training=TrainingConfig(
                lr=5e-5,
                batch_size=64,
                num_epochs=75,
            ),
            loss=LossConfig(
                loss_type="focal",
                focal_alpha=0.80,
                focal_gamma=2.5,
            ),
            data=DataConfig(),
        )
```

### Threshold Tuning

The evaluation script automatically finds the optimal threshold on validation data. You can experiment:

```bash
# Try different thresholds for submission
python -m deepset.evaluate --checkpoint checkpoints/best_model.pt --generate-submission --threshold 0.2 --output sub_0.2.csv
python -m deepset.evaluate --checkpoint checkpoints/best_model.pt --generate-submission --threshold 0.3 --output sub_0.3.csv
python -m deepset.evaluate --checkpoint checkpoints/best_model.pt --generate-submission --threshold 0.4 --output sub_0.4.csv
```

### Pooling Strategies

Deep Sets uses permutation-invariant pooling. Available options:
- `mean`: Average pooling (default, stable)
- `sum`: Sum pooling (sensitive to set size)
- `max`: Max pooling (focuses on extremes)

Change in config:
```python
ModelConfig(pooling="max")
```

### Loss Functions

Two loss functions are available:

#### Focal Loss (recommended for class imbalance)
```python
LossConfig(
    loss_type="focal",
    focal_alpha=0.75,  # Weight for positive class
    focal_gamma=2.0,   # Focusing parameter
)
```

#### Weighted BCE
```python
LossConfig(
    loss_type="bce",
    use_class_weights=True,
    pos_weight_scale=3.0,
)
```

## Monitoring Training

Training outputs:
- Per-epoch metrics (loss, F1, precision, recall)
- Best threshold on validation set
- Checkpoint saving

Example output:
```
Epoch 10/50
  Train Loss: 0.0234
  Val Loss: 0.0256
  Val F1: 0.3421
  Val Precision: 0.2567
  Val Recall: 0.4982
  Best Threshold: 0.32
  ✓ New best model saved! (F1: 0.3421)
```

## Checkpoints

Checkpoints are saved to `checkpoints/`:
- `best_model.pt`: Best model by validation F1
- `checkpoint_epoch_N.pt`: Regular checkpoints every N epochs

Checkpoint contains:
- Model weights
- Optimizer state
- Scheduler state
- Best threshold
- Best F1 score
- Configuration

## Submission Format

Generated CSV has format:
```csv
room_id,operations
0,12 45 67 89
1,
2,23 56 78
```

Where:
- Each row is a room
- `operations` is space-separated operation codes
- Empty string for rooms with no operations

## Tips for Best Results

1. **Start with default config**: It's well-tuned for this problem
2. **Monitor validation F1**: This correlates well with competition score
3. **Try different thresholds**: The optimal threshold can vary
4. **Use focal loss**: Better for extreme class imbalance
5. **Check for overfitting**: If val loss increases while train loss decreases
6. **Ensemble predictions**: Average multiple models/thresholds
7. **Use strong config for final**: Higher capacity can help

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```python
TrainingConfig(batch_size=16)
```

### NaN Loss
Already handled with gradient clipping, but you can adjust:
```python
TrainingConfig(gradient_clip=0.5)
```

### Low Recall
Decrease threshold:
```bash
python -m deepset.evaluate --threshold 0.2
```

### Low Precision
Increase threshold:
```bash
python -m deepset.evaluate --threshold 0.4
```

### Model Not Learning
- Check data loading (run `test_pipeline.py`)
- Try different learning rate
- Try different loss function
- Check for bugs in data preprocessing

## Competition Scoring

The competition uses:
- TP: +1 point
- FP: -0.25 points
- FN: -0.5 points
- Empty room correct: +1 point

False negatives hurt twice as much as false positives, so aim for higher recall.

## Next Steps

1. ✅ Test pipeline: `python -m deepset.test_pipeline`
2. ✅ Debug training: `python -m deepset.train --config debug`
3. ✅ Full training: `python -m deepset.train --config default`
4. ✅ Evaluate: `python -m deepset.evaluate --split val`
5. ✅ Generate submission: `python -m deepset.evaluate --generate-submission`
6. ✅ Submit to Kaggle!
7. ✅ Iterate with different configs/thresholds

Good luck! 🚀
