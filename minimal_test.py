"""
Minimal test to identify NaN source.
"""

import torch
import torch.nn as nn
from models.deep_sets_autoencoder import DeepSetsAutoencoder

# Test model in isolation
device = "cuda"
batch_size = 2
num_clusters = 388
num_companies = 14
num_rooms = 11

print("Creating model...")
model = DeepSetsAutoencoder(
    num_clusters=num_clusters,
    embedding_dim=128,
    hidden_dim=256,
    num_companies=num_companies,
    num_rooms=num_rooms,
    pooling_type="mean",
    use_attention=True,
    dropout=0.1,
    use_auxiliary_head=True
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create simple inputs
X_codes = torch.randint(0, num_clusters, (batch_size, 10)).to(device)
X_mask = torch.ones(batch_size, 10, dtype=torch.bool).to(device)
context = torch.randn(batch_size, 5, num_clusters + num_rooms).to(device)
context_mask = torch.ones(batch_size, 5, dtype=torch.bool).to(device)

insurance_company_one_hot = torch.zeros(batch_size, num_companies).to(device)
insurance_company_one_hot[:, 0] = 1
room_cluster_one_hot = torch.zeros(batch_size, num_rooms).to(device)
room_cluster_one_hot[:, 2] = 1
office_distance = torch.rand(batch_size, 1).to(device)
case_creation_year = torch.rand(batch_size, 1).to(device) * 10 + 2015
case_creation_month = torch.randint(1, 13, (batch_size, 1)).float().to(device)

# Test forward multiple times
print("\nTesting forward pass...")
model.eval()

for i in range(5):
    with torch.no_grad():
        logits, complete_logit = model(
            X_codes=X_codes,
            X_mask=X_mask,
            context=context,
            context_mask=context_mask,
            insurance_company_one_hot=insurance_company_one_hot,
            room_cluster_one_hot=room_cluster_one_hot,
            office_distance=office_distance,
            case_creation_year=case_creation_year,
            case_creation_month=case_creation_month
        )
        
        print(f"Run {i+1}:")
        print(f"  Logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
        print(f"  NaN: {torch.isnan(logits).any()}, Inf: {torch.isinf(logits).any()}")
        
        if torch.isnan(logits).any():
            print("  ERROR: NaN detected!")
            break

# Test with BCE loss
print("\nTesting with BCE loss...")
from utils.losses import WeightedBCELoss

loss_fn = WeightedBCELoss().to(device)
targets = torch.randint(0, 2, (batch_size, num_clusters)).float().to(device)

model.train()
for i in range(3):
    logits, complete_logit = model(
        X_codes=X_codes,
        X_mask=X_mask,
        context=context,
        context_mask=context_mask,
        insurance_company_one_hot=insurance_company_one_hot,
        room_cluster_one_hot=room_cluster_one_hot,
        office_distance=office_distance,
        case_creation_year=case_creation_year,
        case_creation_month=case_creation_month
    )
    
    loss = loss_fn(logits, targets)
    print(f"Iteration {i+1}: loss={loss.item():.4f}, NaN={torch.isnan(loss)}")
    
    if not torch.isnan(loss):
        loss.backward()
        print(f"  Backward successful, grad norm={sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None):.4f}")
        model.zero_grad()
    else:
        print("  ERROR: Loss is NaN!")
        break

print("\n✓ Test complete")
