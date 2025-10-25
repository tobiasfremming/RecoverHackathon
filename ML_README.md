# Deep Sets Autoencoder for Work Operations Prediction

A permutation-invariant set autoencoder based on the Deep Sets architecture for predicting missing work operations in construction/restoration projects.

## 🎯 Overview

This project implements a sophisticated machine learning pipeline for the Recover Hackathon challenge. The goal is to predict which work operations are missing from a room's specification, given:

- **Visible operations**: Operations already listed for the room
- **Context**: Operations from other rooms in the same project (calculus)
- **Metadata**: Project information (insurance company, location, dates, room type)

### Key Innovation: Deep Sets Architecture

Unlike sequence models (RNNs, Transformers), Deep Sets treat operations as **sets** (unordered collections), which is more appropriate for this problem since operation order doesn't matter.

**Architecture components**:
1. **Operation Embeddings**: Learn dense representations for each of 388 operation types
2. **Set Pooling**: Permutation-invariant aggregation (sum/mean/max)
3. **Context Encoder**: Process operations from other rooms with attention
4. **Metadata Encoder**: Encode categorical and numeric project features
5. **Decoder MLP**: Predict missing operations with sigmoid outputs
6. **Auxiliary Head**: Predict if room is complete (no missing operations)

## 📁 Project Structure

```
recover-hackathon/
├── models/
│   ├── __init__.py
│   └── deep_sets_autoencoder.py    # Main model architecture
├── config/
│   ├── __init__.py
│   └── train_config.py              # All hyperparameters and configs
├── utils/
│   ├── __init__.py
│   ├── features.py                  # Feature engineering utilities
│   ├── losses.py                    # Custom loss functions (Focal, Weighted BCE)
│   └── evaluation.py                # Evaluation metrics and analysis
├── dataset/                         # Dataset classes (provided)
│   ├── hackathon.py
│   ├── work_operations.py
│   ├── metadata.py
│   ├── collate.py
│   └── base.py
├── metrics/
│   └── score.py                     # Custom evaluation metric
├── experiments/
│   └── model_training.ipynb         # Interactive training notebook
├── train.py                         # Main training script
├── inference.py                     # Inference and submission script
├── checkpoints/                     # Model checkpoints (created during training)
├── logs/                            # Training logs (created during training)
└── submissions/                     # Competition submissions
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure PyTorch is installed with CUDA support (if using GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Training

**Quick experiment** (10 epochs, smaller batch):
```bash
python train.py --config fast --epochs 10 --batch_size 64
```

**Default training** (50 epochs):
```bash
python train.py --config default
```

**Production training** (100 epochs, larger model):
```bash
python train.py --config production --device cuda
```

**Custom training**:
```bash
python train.py --epochs 30 --batch_size 32 --device cuda
```

### 3. Inference

**Generate submission** (with threshold tuning):
```bash
python inference.py \\
    --checkpoint checkpoints/checkpoint_best.pt \\
    --tune_threshold \\
    --save_predictions
```

**Quick prediction** (fixed threshold):
```bash
python inference.py \\
    --checkpoint checkpoints/checkpoint_best.pt \\
    --threshold 0.5
```

## 📊 Model Architecture Details

### Input Processing

1. **Visible Operations** (X):
   - One-hot encoded vector of length 388
   - Converted to operation codes, then embedded
   - Example: `[1, 5, 23, 145]` → embeddings of shape `(4, embedding_dim)`

2. **Context** (Calculus):
   - List of dictionaries, each containing:
     - Operations from another room (one-hot)
     - Room cluster (one-hot)
   - Processed through attention or pooling

3. **Metadata**:
   - `insurance_company`: One-hot (14 companies)
   - `room_cluster`: One-hot (11 room types)
   - `office_distance`: Normalized numeric
   - `case_creation_year`: Normalized numeric
   - `case_creation_month`: Cyclical encoding (sin/cos)

### Forward Pass

```
X_codes → Embedding → Pooling → h_target (embedding_dim)
Context → Projection → Attention → h_context (embedding_dim)
Metadata → MLP → h_meta (64)

Combined = concat(h_target, h_context, h_meta)
Predictions = Decoder_MLP(Combined) → sigmoid(388)
Complete = Auxiliary_Head(Combined) → sigmoid(1)
```

### Loss Function

**Main Loss**: Binary Cross-Entropy (BCE) with class weights
```python
BCE_weighted = BCE(predictions, targets) * class_weights
```

**Alternative**: Focal Loss for hard examples
```python
FL = -alpha * (1 - p_t)^gamma * log(p_t)
```

**Auxiliary Loss**: Predict room completeness
```python
is_complete = (Y.sum() == 0)  # No missing operations
aux_loss = BCE(complete_pred, is_complete)
```

**Total Loss**:
```python
total_loss = main_loss + 0.1 * aux_loss
```

## 🔧 Configuration

All hyperparameters are in `config/train_config.py`. Key configs:

### Model Hyperparameters
```python
embedding_dim = 128        # Operation embedding dimension
hidden_dim = 256          # Decoder hidden dimension
pooling_type = "mean"     # Set pooling: mean/sum/max
use_attention = True      # Attention for context pooling
dropout = 0.2             # Dropout probability
```

### Training Hyperparameters
```python
batch_size = 32
num_epochs = 50
learning_rate = 1e-3
weight_decay = 1e-4
grad_clip = 1.0
scheduler_type = "cosine"
```

### Loss Configuration
```python
use_focal_loss = False       # Use Focal Loss vs BCE
focal_gamma = 2.0            # Focal loss focusing parameter
use_class_weights = True     # Weight rare operations
auxiliary_loss_weight = 0.1  # Weight for auxiliary head
```

### Inference Configuration
```python
default_threshold = 0.5
tune_threshold = True        # Optimize threshold on validation
threshold_search_range = (0.1, 0.9)
threshold_search_steps = 41
```

## 📈 Evaluation Metrics

### Primary Metric: Room Score

Custom metric from `metrics/score.py`:
- **True Positive**: +1.0
- **False Positive**: -0.25
- **False Negative**: -0.5
- **Empty room correct**: +1.0

Normalized to range [0, 1] relative to dummy baseline.

### Additional Metrics

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **Hamming Loss**: Fraction of incorrect labels
- **Exact Match**: Fraction of perfectly predicted samples

## 🔬 Experiments and Analysis

### Interactive Notebook

Use `experiments/model_training.ipynb` for:
- Quick experimentation
- Threshold optimization visualization
- Embedding analysis (PCA/t-SNE)
- Error pattern analysis
- Per-operation performance

### Hyperparameter Tuning

Suggested parameters to tune:

**High impact**:
- `embedding_dim`: [64, 128, 256]
- `learning_rate`: [1e-4, 5e-4, 1e-3]
- `use_focal_loss`: [True, False]
- `pooling_type`: ['mean', 'sum', 'max']

**Medium impact**:
- `hidden_dim`: [128, 256, 512]
- `dropout`: [0.1, 0.2, 0.3]
- `batch_size`: [16, 32, 64]

**Low impact**:
- `auxiliary_loss_weight`: [0.05, 0.1, 0.2]
- `focal_gamma`: [1.0, 2.0, 3.0]

## 📝 Training Tips

### For Fast Iterations
1. Use `--config fast` (10 epochs, batch 64)
2. Reduce model size: `embedding_dim=64, hidden_dim=128`
3. Use fewer workers: `num_workers=2`

### For Best Performance
1. Use `--config production` (100 epochs, larger model)
2. Enable focal loss for imbalanced data
3. Tune threshold on full validation set
4. Ensemble multiple models with different seeds

### Avoiding Overfitting
- Use dropout (0.2-0.3)
- Weight decay (1e-4)
- Early stopping (patience=10)
- Data augmentation via sampling strategies

### Handling Class Imbalance
- Enable class weights from tickets.csv
- Use focal loss (gamma=2.0)
- Adjust `pos_weight_scale`
- Use balanced sampling

## 🐛 Debugging

### Model not learning?
- Check learning rate (try 1e-4 or 5e-3)
- Verify loss is decreasing
- Check gradient norms (add gradient clipping)
- Ensure data is shuffled

### Poor validation performance?
- Tune threshold on validation set
- Check for overfitting (train vs val loss)
- Try focal loss or adjust class weights
- Increase dropout or weight decay

### Out of memory?
- Reduce batch size
- Reduce embedding_dim or hidden_dim
- Use gradient accumulation
- Use mixed precision training (fp16)

## 📦 Checkpoints

Checkpoints are saved in `checkpoints/` directory:
- `checkpoint_best.pt`: Best validation score
- `checkpoint_latest.pt`: Most recent epoch
- `checkpoint_epoch_N.pt`: Periodic saves

Each checkpoint contains:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'metrics': dict,
    'config': ExperimentConfig
}
```

## 🎯 Next Steps

### Short-term Improvements
1. ✅ Implement proper metadata integration in train.py
2. ✅ Add wandb logging support
3. ✅ Implement mixed precision training
4. ✅ Add model ensembling script

### Medium-term Enhancements
1. Per-operation threshold tuning
2. Curriculum learning (easy → hard samples)
3. Self-supervised pre-training on operations
4. Graph neural network for room relationships

### Advanced Features
1. Transformer-based set encoder
2. Contrastive learning for embeddings
3. Multi-task learning (predict costs, durations)
4. Active learning for labeling

## 📚 References

- **Deep Sets**: [Zaheer et al., 2017](https://arxiv.org/abs/1703.06114)
- **Focal Loss**: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- **Set2Set**: [Vinyals et al., 2015](https://arxiv.org/abs/1511.06391)

## 🤝 Contributing

For questions or improvements, please:
1. Check existing issues
2. Create detailed bug reports
3. Submit pull requests with tests

## 📄 License

This project is part of the Recover Hackathon competition.

---

**Happy Training! 🚀**
