# 🚀 Quick Start Guide - Deep Sets ML Pipeline

## ⚡ 5-Minute Setup

### Step 1: Verify Setup
```powershell
# Run the test script
python test_model.py
```

You should see:
```
✓ Model created successfully!
✓ Total parameters: 344,518
✓ All tests passed!
```

### Step 2: Quick Training Test (10 minutes)
```powershell
python train.py --config fast --epochs 5
```

This runs a quick 5-epoch test to verify training works.

### Step 3: Generate Submission
```powershell
python inference.py --checkpoint checkpoints/checkpoint_best.pt --tune_threshold
```

Find your submission in `submissions/submission_YYYYMMDD_HHMMSS.csv`

---

## 📖 Complete Workflow

### 1. Full Training (2-4 hours)

**Option A: Default Configuration**
```powershell
python train.py --config default
```
- 50 epochs
- Batch size: 32
- Embedding dim: 128
- Expected validation score: ~0.6-0.7

**Option B: Production Configuration**
```powershell
python train.py --config production
```
- 100 epochs
- Larger model (embedding_dim=256, hidden_dim=512)
- Focal loss enabled
- Expected validation score: ~0.7-0.8

**Option C: Custom Training**
```powershell
python train.py --epochs 30 --batch_size 64 --device cuda
```

### 2. Monitor Training

During training, you'll see:
```
Epoch 1/50
Train metrics:
  train_loss: 0.6234
  train_main_loss: 0.5821
  train_aux_loss: 0.4132

Validation metrics:
  val_loss: 0.5892
  precision: 0.6543
  recall: 0.5987
  f1: 0.6253
  room_score: 0.6789
  
✓ New best validation score: 0.6789
```

Checkpoints are saved to `checkpoints/`:
- `checkpoint_best.pt` - Best validation score
- `checkpoint_latest.pt` - Most recent epoch
- `checkpoint_epoch_N.pt` - Periodic saves

### 3. Inference & Submission

**With Threshold Tuning (Recommended)**
```powershell
python inference.py `
    --checkpoint checkpoints/checkpoint_best.pt `
    --tune_threshold `
    --save_predictions
```

Output:
```
Tuning threshold on validation set...
Best threshold: 0.423 (score: 0.7234)

Running inference on test set...
Predictions shape: torch.Size([18299, 388])

Creating submission file...
✓ Submission saved to: submissions/submission_20240115_143022.csv
✓ Raw predictions saved to: submissions/test_predictions.pt
```

**Quick Inference (Fixed Threshold)**
```powershell
python inference.py `
    --checkpoint checkpoints/checkpoint_best.pt `
    --threshold 0.5
```

---

## 🔧 Configuration Options

### Model Architecture
Edit `config/train_config.py` → `ModelConfig`:

```python
embedding_dim = 128        # Operation embedding size (64, 128, 256)
hidden_dim = 256          # Decoder hidden size (128, 256, 512)
pooling_type = "mean"     # Set pooling: mean, sum, max
use_attention = True      # Attention for context pooling
dropout = 0.2             # Dropout rate (0.1, 0.2, 0.3)
use_auxiliary_head = True # Predict room completeness
```

### Training Parameters
Edit `config/train_config.py` → `TrainingConfig`:

```python
batch_size = 32              # Batch size (16, 32, 64)
num_epochs = 50              # Training epochs
learning_rate = 1e-3         # Initial LR (1e-4, 5e-4, 1e-3)
weight_decay = 1e-4          # L2 regularization
grad_clip = 1.0              # Gradient clipping threshold
scheduler_type = "cosine"    # LR schedule: cosine, plateau, step
early_stopping_patience = 10 # Early stopping patience
```

### Loss Function
Edit `config/train_config.py` → `LossConfig`:

```python
use_focal_loss = False          # Use Focal Loss vs BCE
focal_gamma = 2.0               # Focal loss gamma (1.0, 2.0, 3.0)
use_class_weights = True        # Weight by operation frequency
auxiliary_loss_weight = 0.1     # Weight for auxiliary task
```

---

## 📊 Understanding Outputs

### Training Logs

**Metrics Explained**:
- `train_loss`: Total loss (main + auxiliary)
- `train_main_loss`: Operation prediction loss
- `train_aux_loss`: Room completeness prediction loss
- `val_loss`: Validation total loss
- `precision`: TP / (TP + FP) - how many predictions are correct
- `recall`: TP / (TP + FN) - how many actual operations found
- `f1`: Harmonic mean of precision and recall
- `room_score`: Custom metric (TP=+1, FP=-0.25, FN=-0.5)

**Good Signs**:
- ✓ Training loss decreasing steadily
- ✓ Validation loss following training loss
- ✓ Room score improving
- ✓ Checkpoints being saved

**Warning Signs**:
- ⚠ Training loss decreasing but validation loss increasing (overfitting)
- ⚠ Loss not decreasing (learning rate too high/low)
- ⚠ Room score not improving (threshold issue)

### Checkpoints

Each checkpoint contains:
```python
checkpoint = {
    'epoch': 23,                    # Training epoch
    'model_state_dict': {...},      # Model weights
    'optimizer_state_dict': {...},  # Optimizer state
    'scheduler_state_dict': {...},  # LR scheduler state
    'metrics': {                    # Performance metrics
        'train_loss': 0.5234,
        'val_room_score': 0.7123,
        ...
    },
    'config': ExperimentConfig(...) # Full configuration
}
```

Load checkpoint:
```python
checkpoint = torch.load('checkpoints/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
best_score = checkpoint['metrics']['val_room_score']
```

---

## 🎯 Hyperparameter Tuning

### Priority 1: High Impact

**Embedding Dimension**
```python
embedding_dim = [64, 128, 256]  # Try all three
```
- 64: Faster, less memory, may underfit
- 128: Balanced (default)
- 256: Better performance, more memory

**Pooling Type**
```python
pooling_type = ["mean", "sum", "max"]
```
- mean: Most stable (default)
- sum: Sensitive to set size
- max: Captures strongest signals

**Loss Function**
```python
use_focal_loss = [True, False]
focal_gamma = [1.0, 2.0, 3.0]
```
- Focal loss helps with class imbalance
- Higher gamma = focus more on hard examples

### Priority 2: Medium Impact

**Learning Rate**
```python
learning_rate = [1e-4, 5e-4, 1e-3, 5e-3]
```
Test with quick runs (10 epochs).

**Dropout**
```python
dropout = [0.1, 0.2, 0.3]
```
- Lower: Less regularization, may overfit
- Higher: More regularization, may underfit

**Batch Size**
```python
batch_size = [16, 32, 64]
```
- Smaller: Noisier gradients, slower
- Larger: Smoother gradients, more memory

### Priority 3: Fine-tuning

**Hidden Dimension**
```python
hidden_dim = [128, 256, 512]
```

**Auxiliary Loss Weight**
```python
auxiliary_loss_weight = [0.05, 0.1, 0.2]
```

**Scheduler Parameters**
```python
T_max = [50, 100]  # For cosine annealing
eta_min = [1e-6, 1e-5]
```

### Tuning Strategy

1. **Coarse Search** (1-2 hours)
   - embedding_dim: [64, 128, 256]
   - pooling_type: [mean, sum]
   - use_focal_loss: [True, False]
   - Quick runs: 10 epochs each

2. **Refined Search** (2-4 hours)
   - Best config from coarse search
   - learning_rate: [1e-4, 5e-4, 1e-3]
   - dropout: [0.1, 0.2, 0.3]
   - 30 epochs each

3. **Final Training** (4-8 hours)
   - Best overall config
   - 100 epochs with early stopping
   - Full production settings

---

## 🐛 Troubleshooting

### "CUDA out of memory"

**Solution 1: Reduce batch size**
```powershell
python train.py --batch_size 16
```

**Solution 2: Reduce model size**
```python
# config/train_config.py
embedding_dim = 64
hidden_dim = 128
```

**Solution 3: Use CPU**
```powershell
python train.py --device cpu
```

### "Loss is NaN"

**Possible causes**:
- Learning rate too high
- Gradient explosion

**Solutions**:
```python
learning_rate = 1e-4  # Lower LR
grad_clip = 0.5       # Stronger clipping
```

### "Model not improving"

**Check 1: Learning rate**
```python
learning_rate = [1e-4, 1e-3, 1e-2]  # Try different values
```

**Check 2: Class weights**
```python
use_class_weights = True
pos_weight_scale = 2.0  # Increase if rare ops ignored
```

**Check 3: Threshold**
```powershell
# After training, tune threshold
python inference.py --checkpoint ... --tune_threshold
```

### "Validation worse than training"

**Overfitting!** Solutions:
```python
dropout = 0.3              # Increase dropout
weight_decay = 1e-3        # Increase regularization
early_stopping_patience = 5  # Stop earlier
```

### "Training too slow"

**Speed up**:
```python
num_workers = 8      # More data loading workers
batch_size = 64      # Larger batches
pin_memory = True    # Faster GPU transfer
```

### "Out of disk space"

Checkpoints can be large! Clean up:
```powershell
# Remove old checkpoints
Remove-Item checkpoints/checkpoint_epoch_*.pt
# Keep only best and latest
```

---

## 📈 Advanced Usage

### Interactive Notebook

```powershell
jupyter notebook experiments/model_training.ipynb
```

Use for:
- Quick experiments
- Embedding visualization
- Error analysis
- Threshold optimization plots

### Model Ensembling

Train multiple models with different seeds:
```powershell
python train.py --config production --seed 42
python train.py --config production --seed 123
python train.py --config production --seed 456
```

Average predictions:
```python
preds1 = torch.load('submissions/test_predictions_seed42.pt')
preds2 = torch.load('submissions/test_predictions_seed123.pt')
preds3 = torch.load('submissions/test_predictions_seed456.pt')

ensemble_preds = (preds1 + preds2 + preds3) / 3
```

### Custom Features

Add new metadata features:

1. **Update dataset**:
```python
# dataset/hackathon.py
def __getitem__(self, idx):
    ...
    return {
        ...
        'custom_feature': compute_custom_feature(...)
    }
```

2. **Update model**:
```python
# models/deep_sets_autoencoder.py
class MetadataEncoder(nn.Module):
    def __init__(self, ..., custom_dim=10):
        input_dim = ... + custom_dim
        ...
```

3. **Update training**:
```python
# train.py
custom_feature = batch['custom_feature']
predictions, _ = model(..., custom_feature=custom_feature)
```

---

## 📝 Checklist Before Competition Submission

- [ ] Trained model with best hyperparameters
- [ ] Validated on validation set (score > 0.7)
- [ ] Tuned threshold on validation set
- [ ] Generated submission file
- [ ] Checked submission format (18,299 rows, 389 columns)
- [ ] Verified no duplicate IDs
- [ ] Backed up best checkpoint
- [ ] Saved training logs
- [ ] Documented final configuration

---

## 🎓 Tips for Best Results

### Data
1. **Understand the data**
   - Explore `data/` files in notebooks
   - Check co-occurrence patterns
   - Identify room-specific operations

2. **Feature engineering**
   - Cyclical month encoding (sin/cos)
   - Normalize numeric features
   - One-hot encode categoricals

### Model
1. **Start simple**
   - Baseline: embedding_dim=64, mean pooling
   - Gradually increase complexity

2. **Use attention**
   - Context rooms provide valuable signals
   - Attention learns importance

3. **Auxiliary task**
   - "Room complete" helps regularization
   - Improves calibration

### Training
1. **Monitor metrics**
   - Watch room_score on validation
   - Check precision/recall balance
   - Early stopping prevents overfitting

2. **Tune threshold**
   - Default 0.5 often suboptimal
   - Grid search on validation
   - FN penalty > FP penalty → lower threshold

3. **Class weights**
   - Essential for rare operations
   - Use tickets.csv frequencies
   - Prevents "predict nothing" collapse

### Inference
1. **Threshold optimization**
   - Always tune on validation
   - Can boost score by 0.05-0.10
   - Per-operation thresholds even better

2. **Ensembling**
   - Train 3-5 models with different seeds
   - Average predictions
   - Typically +0.02-0.05 improvement

3. **Post-processing**
   - Remove impossible combinations
   - Enforce room-specific constraints
   - Use co-occurrence rules

---

## 📚 Further Reading

**Documentation**:
- `ML_README.md` - Comprehensive ML guide
- `IMPLEMENTATION_SUMMARY.md` - Full technical details
- `README.md` - Project overview

**Code**:
- `models/` - Model architecture
- `config/` - All hyperparameters
- `utils/` - Feature engineering, losses, evaluation
- `train.py` - Training script
- `inference.py` - Prediction script

**Examples**:
- `experiments/model_training.ipynb` - Interactive notebook
- `test_model.py` - Architecture test

---

## 🚀 Ready to Start!

```powershell
# Test everything works
python test_model.py

# Quick training test
python train.py --config fast --epochs 5

# Full training
python train.py --config default

# Generate submission
python inference.py --checkpoint checkpoints/checkpoint_best.pt --tune_threshold
```

**Good luck! 🏆**

---

*For questions or issues, check the troubleshooting section above or review the implementation summary.*
