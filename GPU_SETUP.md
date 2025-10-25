# GPU Setup Guide for Windows

## Current Issue

Your PyTorch installation is **CPU-only**. To use your NVIDIA GPU for training, you need to install PyTorch with CUDA support.

## Check Your GPU

First, verify you have an NVIDIA GPU:

```powershell
nvidia-smi
```

You should see your GPU model and driver version.

## Install PyTorch with CUDA

### Option 1: Using uv (Recommended)

```powershell
# Uninstall CPU-only PyTorch
uv pip uninstall torch torchvision torchaudio

# Install PyTorch with CUDA 12.1 (adjust version if needed)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option 2: Using pip directly

```powershell
# Activate your virtual environment first
.\.venv\Scripts\Activate.ps1

# Uninstall CPU-only PyTorch
pip uninstall torch torchvision torchaudio

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option 3: For CUDA 11.8

If your GPU driver doesn't support CUDA 12.1, use CUDA 11.8:

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verify Installation

After installation, verify CUDA is working:

```powershell
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

You should see:
```
CUDA available: True
CUDA version: 12.1 (or 11.8)
GPU: NVIDIA GeForce RTX ... (your GPU model)
```

## Performance Comparison

**CPU Training** (current):
- 5 epochs: ~20-30 minutes
- 50 epochs: ~3-5 hours

**GPU Training** (with CUDA):
- 5 epochs: ~2-5 minutes ⚡
- 50 epochs: ~30-60 minutes ⚡

**Speed improvement: ~10-20x faster!**

## After Installing CUDA PyTorch

Run training with GPU:

```powershell
# Quick test
uv run python train.py --config fast --epochs 5 --device cuda

# Full training
uv run python train.py --config default --device cuda

# Production training
uv run python train.py --config production --device cuda
```

## Troubleshooting

### "CUDA out of memory"

If you get out of memory errors:

```powershell
# Reduce batch size
uv run python train.py --config default --batch_size 16 --device cuda

# Or use smaller model
# Edit config/train_config.py:
# embedding_dim = 64
# hidden_dim = 128
```

### "CUDA driver version is insufficient"

Update your NVIDIA drivers:
1. Go to https://www.nvidia.com/Download/index.aspx
2. Download latest driver for your GPU
3. Install and restart

### "RuntimeError: No CUDA GPUs are available"

Check if GPU is visible:
```powershell
nvidia-smi
```

If GPU doesn't show up, try:
1. Restart computer
2. Update NVIDIA drivers
3. Check if GPU is disabled in Device Manager

## Current Status (CPU-only)

✅ **Fixes Applied**:
- Dtype mismatch fixed (Y is now float for BCE loss)
- Windows multiprocessing serialization fixed
- Training should work on CPU now

⚠ **Performance Note**:
Training on CPU will be **10-20x slower** than GPU. For the hackathon:
- **Quick testing**: Use CPU with `--config fast --epochs 5`
- **Final submission**: Install CUDA PyTorch and use GPU

## Next Steps

1. **Option A: Continue with CPU** (slower but works)
   ```powershell
   # Training is currently running, wait for it to complete
   # It will take ~15-20 minutes for 2 epochs
   ```

2. **Option B: Install CUDA and switch to GPU** (recommended)
   ```powershell
   # Stop current training (Ctrl+C)
   # Install PyTorch with CUDA (see above)
   # Run with GPU
   uv run python train.py --config fast --epochs 5 --device cuda
   ```

---

**Recommendation**: Install PyTorch with CUDA for significantly faster training! 🚀
