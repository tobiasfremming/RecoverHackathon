"""
Quick test script to verify Deep Sets model architecture.

This script creates a model instance and runs a forward pass with dummy data
to verify everything is working correctly before training.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
from models.deep_sets_autoencoder import DeepSetsAutoencoder
from utils.losses import create_loss_function, compute_room_complete_targets
from config.train_config import get_default_config

def test_model():
    """Test model creation and forward pass."""
    
    print("="*60)
    print("Testing Deep Sets Autoencoder Architecture")
    print("="*60)
    
    # Configuration
    config = get_default_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Model parameters
    batch_size = 4
    num_clusters = 388
    num_companies = 14
    num_rooms = 11
    max_ops = 10
    max_context_rooms = 5
    feature_dim = num_clusters + num_rooms
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num operations: {num_clusters}")
    print(f"  Embedding dim: {config.model.embedding_dim}")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    
    # Create model
    print("\n1. Creating model...")
    model = DeepSetsAutoencoder(
        num_clusters=num_clusters,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        num_companies=num_companies,
        num_rooms=num_rooms,
        pooling_type=config.model.pooling_type,
        use_attention=config.model.use_attention,
        dropout=config.model.dropout,
        use_auxiliary_head=config.model.use_auxiliary_head
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✓ Model created successfully!")
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    print(f"   ✓ Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Create dummy inputs
    print("\n2. Creating dummy inputs...")
    X_codes = torch.randint(0, num_clusters, (batch_size, max_ops)).to(device)
    X_mask = torch.ones(batch_size, max_ops, dtype=torch.bool).to(device)
    X_mask[0, 5:] = False  # First sample has only 5 operations
    X_mask[1, 7:] = False  # Second sample has 7 operations
    
    context = torch.randn(batch_size, max_context_rooms, feature_dim).to(device)
    context_mask = torch.ones(batch_size, max_context_rooms, dtype=torch.bool).to(device)
    context_mask[1, 3:] = False  # Second sample has only 3 context rooms
    context_mask[3, 4:] = False  # Fourth sample has 4 context rooms
    
    insurance_company_one_hot = torch.zeros(batch_size, num_companies).to(device)
    insurance_company_one_hot[:, 0] = 1  # All from company 0
    
    room_cluster_one_hot = torch.zeros(batch_size, num_rooms).to(device)
    room_cluster_one_hot[0, 2] = 1  # Kitchen
    room_cluster_one_hot[1, 3] = 1  # Bathroom
    room_cluster_one_hot[2, 1] = 1  # Living room
    room_cluster_one_hot[3, 4] = 1  # Bedroom
    
    office_distance = torch.rand(batch_size, 1).to(device) * 100
    case_creation_year = torch.rand(batch_size, 1).to(device) * 10 + 2015
    case_creation_month = torch.randint(1, 13, (batch_size, 1)).float().to(device)
    
    print(f"   ✓ Inputs created:")
    print(f"     - X_codes: {X_codes.shape}")
    print(f"     - X_mask: {X_mask.shape}, valid ops: {X_mask.sum(dim=1).tolist()}")
    print(f"     - context: {context.shape}")
    print(f"     - context_mask: {context_mask.shape}, valid rooms: {context_mask.sum(dim=1).tolist()}")
    print(f"     - metadata: insurance, room, distance, year, month")
    
    # Forward pass
    print("\n3. Running forward pass...")
    model.eval()
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
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(logits)
        complete_pred = torch.sigmoid(complete_logit) if complete_logit is not None else None
    
    print(f"   ✓ Forward pass successful!")
    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   ✓ Complete prediction shape: {complete_pred.shape}")
    print(f"   ✓ Complete prediction range: [{complete_pred.min():.4f}, {complete_pred.max():.4f}]")
    
    # Test loss computation
    print("\n4. Testing loss computation...")
    Y = torch.randint(0, 2, (batch_size, num_clusters)).float().to(device)
    Y[0, :] = 0  # First sample has no missing operations (complete room)
    complete_target = compute_room_complete_targets(Y)
    
    # Create loss function
    from utils.losses import create_loss_function
    loss_fn = create_loss_function(config.loss, class_weights=None, device=device)
    
    # Need to recreate logits for loss
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
    
    total_loss, main_loss, aux_loss = loss_fn(
        logits, Y, complete_logit, complete_target
    )
    
    print(f"   ✓ Loss computation successful!")
    print(f"   ✓ Total loss: {total_loss.item():.4f}")
    print(f"   ✓ Main loss: {main_loss.item():.4f}")
    print(f"   ✓ Auxiliary loss: {aux_loss.item():.4f}")
    print(f"   ✓ Complete targets: {complete_target.squeeze().tolist()}")
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
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
    
    total_loss, main_loss, aux_loss = loss_fn(
        logits, Y, complete_logit, complete_target
    )
    
    optimizer.zero_grad()
    total_loss.backward()
    
    # Check gradients
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"   ✓ Backward pass successful!")
    print(f"   ✓ Gradient norm: {grad_norm:.4f}")
    
    optimizer.step()
    print(f"   ✓ Optimizer step successful!")
    
    # Test evaluation
    print("\n6. Testing evaluation metrics...")
    from utils.evaluation import evaluate_predictions
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(
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
        predictions = torch.sigmoid(logits)
    
    metrics = evaluate_predictions(predictions, Y, threshold=0.5)
    
    print(f"   ✓ Evaluation successful!")
    print(f"   ✓ Metrics computed:")
    for key, value in list(metrics.items())[:5]:
        if isinstance(value, float):
            print(f"       - {key}: {value:.4f}")
        else:
            print(f"       - {key}: {value}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nModel is ready for training!")
    print("\nNext steps:")
    print("  1. Run: python train.py --config fast --epochs 5")
    print("  2. Check: checkpoints/ directory for saved models")
    print("  3. Evaluate: python inference.py --checkpoint checkpoints/checkpoint_best.pt")
    print("\n" + "="*60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_model()
        if success:
            print("\n✓ Test completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
