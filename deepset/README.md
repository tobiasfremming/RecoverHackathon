# Deep Sets Implementation for Recover Hackathon

A clean, modular implementation of Deep Sets architecture for predicting missing work operations.

## 📁 Structure

```
deepset/
├── model.py          # Deep Sets model architecture
├── dataset.py        # Data loading and preprocessing
├── train.py          # Training loop
├── config.py         # Configuration and hyperparameters
├── utils.py          # Helper functions
└── README.md         # This file
```

## 🎯 Task

Predict which work operations are missing from a room specification, given:
- Visible operations in the room (variable-length set)
- Context from other rooms in the same project (calculus)
- Project metadata (insurance company, location, etc.)

## 🏗️ Architecture

**Deep Sets** = Permutation-invariant neural network for sets

```
Input Operations → Embedding → Pooling → MLP → Predictions
                                  ↑
                              Context Set
                                  ↑  
                              Metadata
```

### Key Components:

1. **Operation Embedding**: Maps 388 operation codes to dense vectors
2. **Set Pooling**: Permutation-invariant aggregation (mean/sum/max)
3. **Context Encoder**: Processes other rooms in project
4. **Metadata Encoder**: Encodes insurance company, room type, etc.
5. **Decoder**: Predicts missing operations (388-dim binary output)

## 🚀 Quick Start

### Training

```bash
# Basic training
python -m deepset.train

# With custom config
python -m deepset.train --epochs 50 --batch_size 64 --lr 1e-4
```

### Evaluation

```bash
python -m deepset.train --eval_only --checkpoint deepset/checkpoints/best.pt
```

## ⚙️ Configuration

Edit `config.py` to adjust:
- Model architecture (embedding dims, hidden sizes)
- Training hyperparameters (LR, batch size, epochs)
- Loss weights (focal loss, class balancing)
- Data augmentation strategies

## 📊 Metrics

- **F1 Score**: Harmonic mean of precision/recall
- **Room Score**: Competition metric (TP - 0.25×FP - 0.5×FN)
- **Precision/Recall**: Per-operation performance

## 🎓 References

- [Deep Sets paper](https://arxiv.org/abs/1703.06114)
- [Focal Loss](https://arxiv.org/abs/1708.02002)
