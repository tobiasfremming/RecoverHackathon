# Training and Evaluation Guide

## Training

### Quick Training (Debug)
```bash
uv run python -m deepset.train --config debug
```
2 epochs, batch size 8, fast iteration for testing

### Standard Training (Recommended)
```bash
uv run python -m deepset.train --config optimized --loss competition_focal
```
50 epochs, batch size 32, competition-aligned focal loss, 2-3 hours on GPU

### Strong Model Training
```bash
uv run python -m deepset.train --config strong
```
100 epochs, larger model (256d/512d), 8-10 hours on GPU

### Resume Training
```bash
uv run python -m deepset.train --config optimized --resume checkpoints/best_model.pt
```

## Evaluation

### Generate Submission
```bash
uv run python -m deepset.evaluate --checkpoint checkpoints/best_model.pt --generate-submission
```
Creates `submissions/submission_YYYYMMDD_HHMMSS.csv`

### Evaluate on Validation Set
```bash
uv run python -m deepset.evaluate --checkpoint checkpoints/best_model.pt
```
Shows F1, precision, recall, room score metrics

### Generate with Custom Threshold
```bash
uv run python -m deepset.evaluate --checkpoint checkpoints/best_model.pt --generate-submission --threshold 0.50
```

### Generate with Optimizations
```bash
uv run python -m deepset.evaluate --checkpoint checkpoints/best_model.pt --generate-submission --use-dual-threshold --suppress-problematic-ops
```
Applies dual-threshold strategy and per-operation suppression

## Monitoring Training

Best model saved to `checkpoints/best_model.pt` when validation F1 improves

Check progress:
```bash
ls -lh checkpoints/
```

## Configuration Presets

| Preset | Epochs | Batch | Model Size | Time |
|--------|--------|-------|------------|------|
| debug | 2 | 8 | 192d/384d | 5-10 min |
| optimized | 50 | 32 | 192d/384d | 2-3 hours |
| strong | 100 | 32 | 256d/512d | 8-10 hours |
| ultimate | 60 | 28 | 256d/512d | 4-5 hours |

## Loss Functions

| Loss Type | Description |
|-----------|-------------|
| `focal` | Standard focal loss |
| `competition_focal` | Competition-aligned focal loss (recommended) |
| `competition_aligned` | Asymmetric BCE without focal weighting |

Specify with `--loss` flag during training
