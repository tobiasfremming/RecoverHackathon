"""
Quick diagnostic script to test one batch of training.
"""

import torch
import torch.nn as nn
from dataset.hackathon import HackathonDataset
from dataset.collate import collate_fn
from torch.utils.data import DataLoader
from models.deep_sets_autoencoder import DeepSetsAutoencoder
from utils.losses import create_loss_function, compute_room_complete_targets
from config.train_config import get_fast_experiment_config

# Config
config = get_fast_experiment_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load small dataset
print("\nLoading dataset...")
train_dataset = HackathonDataset(
    split="train",
    sampling_strategy=config.sampling.train_sampling_strategies,
    root="data/hackathon-recover-x-cogito"
)

print(f"Dataset size: {len(train_dataset)}")

# Create dataloader with small batch
dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

# Get one batch
print("\nGetting one batch...")
batch = next(iter(dataloader))
print(f"Batch X shape: {batch['X'].shape}")
print(f"Batch Y shape: {batch['Y'].shape}")
print(f"Batch context shape: {batch['context'].shape}")

# Create model
print("\nCreating model...")
model = DeepSetsAutoencoder(
    num_clusters=config.model.num_clusters,
    embedding_dim=config.model.embedding_dim,
    hidden_dim=config.model.hidden_dim,
    num_companies=config.model.num_companies,
    num_rooms=config.model.num_rooms,
    pooling_type=config.model.pooling_type,
    use_attention=config.model.use_attention,
    dropout=config.model.dropout,
    use_auxiliary_head=config.model.use_auxiliary_head
).to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Prepare batch
def prepare_batch(batch, device):
    X = batch["X"].to(device)
    Y = batch["Y"].to(device).float()
    context = batch["context"].to(device)
    
    # Split X into operations and room_cluster
    operations_one_hot = X[:, :-11]
    room_cluster_one_hot = X[:, -11:]
    
    # Convert one-hot to codes for operations
    X_codes = torch.argmax(operations_one_hot, dim=2)
    X_mask = (operations_one_hot.sum(dim=2) > 0)
    
    # Context mask
    context_mask = (context.sum(dim=2) > 0)
    
    return {
        "X_codes": X_codes,
        "X_mask": X_mask,
        "Y": Y,
        "context": context,
        "context_mask": context_mask,
        "room_cluster_one_hot": room_cluster_one_hot[:, 0, :]
    }

print("\nPreparing batch...")
prepared = prepare_batch(batch, device)
print(f"X_codes shape: {prepared['X_codes'].shape}")
print(f"X_codes range: [{prepared['X_codes'].min()}, {prepared['X_codes'].max()}]")
print(f"Y shape: {prepared['Y'].shape}")
print(f"Context shape: {prepared['context'].shape}")

# Create dummy metadata
batch_size = prepared["X_codes"].shape[0]
insurance_company_one_hot = torch.zeros(batch_size, 14).to(device)
insurance_company_one_hot[:, 0] = 1
office_distance = torch.zeros(batch_size, 1).to(device)
case_creation_year = torch.zeros(batch_size, 1).to(device)
case_creation_month = torch.ones(batch_size, 1).to(device) * 6

# Forward pass
print("\nRunning forward pass...")
model.eval()
with torch.no_grad():
    logits, complete_logit = model(
        X_codes=prepared["X_codes"],
        X_mask=prepared["X_mask"],
        context=prepared["context"],
        context_mask=prepared["context_mask"],
        insurance_company_one_hot=insurance_company_one_hot,
        room_cluster_one_hot=prepared["room_cluster_one_hot"],
        office_distance=office_distance,
        case_creation_year=case_creation_year,
        case_creation_month=case_creation_month
    )

print(f"Logits shape: {logits.shape}")
print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}, std={logits.std():.4f}")
print(f"NaN in logits: {torch.isnan(logits).any()}")
print(f"Inf in logits: {torch.isinf(logits).any()}")

# Apply sigmoid
predictions = torch.sigmoid(logits)
print(f"\nPredictions stats: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")

# Test loss
print("\nTesting loss...")
loss_fn = create_loss_function(config.loss, class_weights=None, device=device)

Y = prepared["Y"]
complete_target = compute_room_complete_targets(Y)

total_loss, main_loss, aux_loss = loss_fn(
    logits, Y, complete_logit, complete_target
)

print(f"Total loss: {total_loss.item():.4f}")
print(f"Main loss: {main_loss.item():.4f}")
print(f"Aux loss: {aux_loss.item():.4f}")
print(f"Loss is NaN: {torch.isnan(total_loss)}")

# Test backward
print("\nTesting backward pass...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

logits, complete_logit = model(
    X_codes=prepared["X_codes"],
    X_mask=prepared["X_mask"],
    context=prepared["context"],
    context_mask=prepared["context_mask"],
    insurance_company_one_hot=insurance_company_one_hot,
    room_cluster_one_hot=prepared["room_cluster_one_hot"],
    office_distance=office_distance,
    case_creation_year=case_creation_year,
    case_creation_month=case_creation_month
)

total_loss, main_loss, aux_loss = loss_fn(
    logits, Y, complete_logit, complete_target
)

optimizer.zero_grad()
total_loss.backward()

# Check gradients
grad_norm = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        param_norm = param.grad.data.norm(2).item()
        grad_norm += param_norm ** 2
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf gradient in {name}")

grad_norm = grad_norm ** 0.5
print(f"Gradient norm: {grad_norm:.4f}")

optimizer.step()
print("✓ Optimizer step successful!")

print("\n" + "="*60)
print("Debug complete! Model looks good.")
print("="*60)
