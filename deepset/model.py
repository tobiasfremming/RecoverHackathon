"""
Deep Sets model for work operations prediction.

Based on "Deep Sets" (Zaheer et al., 2017): https://arxiv.org/abs/1703.06114

Key idea: Process variable-length sets in a permutation-invariant way.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DeepSetsModel(nn.Module):
    """
    Deep Sets architecture for predicting missing work operations.
    
    Architecture:
        1. Embed observed operations → embeddings
        2. Pool embeddings → summary vector (permutation-invariant)
        3. Encode context rooms → context vector
        4. Combine target + context → joint representation
        5. Decode → predictions for 388 operations
    """
    
    def __init__(
        self,
        num_operations: int = 388,
        num_rooms: int = 11,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        pooling: str = "mean",
        dropout: float = 0.1,
        use_context: bool = True,
        use_empty_room_head: bool = False,
    ):
        super().__init__()
        
        self.num_operations = num_operations
        self.num_rooms = num_rooms
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.use_context = use_context
        self.use_empty_room_head = use_empty_room_head
        
        # Total feature dimension (operations + room type)
        self.feature_dim = num_operations + num_rooms
        
        # Embedding layer: maps binary features to dense vectors
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        # Context encoder: process other rooms in project
        if use_context:
            self.context_encoder = nn.Sequential(
                nn.Linear(self.feature_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
            decoder_input_dim = embedding_dim * 2  # target + context
        else:
            decoder_input_dim = embedding_dim
        
        # Decoder: predict missing operations
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_operations),
        )
        
        # Empty room classifier (auxiliary task)
        if use_empty_room_head:
            self.empty_room_classifier = nn.Sequential(
                nn.Linear(decoder_input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier/Kaiming initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def pool(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Permutation-invariant pooling.
        
        Args:
            embeddings: (batch, set_size, embedding_dim)
            mask: (batch, set_size) - True for valid elements
        
        Returns:
            pooled: (batch, embedding_dim)
        """
        if mask is not None:
            # Expand mask for broadcasting
            mask = mask.unsqueeze(-1).float()  # (batch, set_size, 1)
            embeddings = embeddings * mask
        
        if self.pooling == "mean":
            if mask is not None:
                # Average only over valid elements
                sum_embeddings = embeddings.sum(dim=1)
                count = mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
                return sum_embeddings / count
            else:
                return embeddings.mean(dim=1)
        
        elif self.pooling == "sum":
            return embeddings.sum(dim=1)
        
        elif self.pooling == "max":
            if mask is not None:
                # Set masked positions to -inf before max pooling
                embeddings = embeddings.masked_fill(~mask.bool(), float('-inf'))
            return embeddings.max(dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def encode_target(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encode target room features.
        
        Args:
            X: (batch, feature_dim) - observed operations + room type
        
        Returns:
            encoded: (batch, embedding_dim)
        """
        # X is already concatenated features (operations + room_type)
        # Just pass through encoder
        return self.feature_encoder(X.float())
    
    def encode_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode context rooms.
        
        Args:
            context: (batch, max_rooms, feature_dim)
            context_mask: (batch, max_rooms) - True for valid rooms
        
        Returns:
            context_vec: (batch, embedding_dim)
        """
        if not self.use_context:
            batch_size = context.shape[0]
            return torch.zeros(batch_size, self.embedding_dim, device=context.device)
        
        # Encode each room
        batch_size, max_rooms, feature_dim = context.shape
        context_flat = context.view(-1, feature_dim)  # (batch*max_rooms, feature_dim)
        context_encoded = self.context_encoder(context_flat.float())  # (batch*max_rooms, embedding_dim)
        context_encoded = context_encoded.view(batch_size, max_rooms, self.embedding_dim)
        
        # Pool across rooms (permutation-invariant)
        context_vec = self.pool(context_encoded, context_mask)
        
        return context_vec
    
    def forward(
        self,
        X: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_empty_room_logits: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            X: (batch, feature_dim) - observed operations + room type
            context: (batch, max_rooms, feature_dim) - context rooms
            context_mask: (batch, max_rooms) - mask for valid context rooms
            return_empty_room_logits: If True, return (operation_logits, empty_room_logits)
        
        Returns:
            logits: (batch, num_operations) - raw predictions (use sigmoid to get probabilities)
            OR
            (logits, empty_room_logits): if return_empty_room_logits=True
        """
        # Encode target room
        target_vec = self.encode_target(X)
        
        # Encode context
        if self.use_context and context is not None:
            context_vec = self.encode_context(context, context_mask)
            # Combine target and context
            combined = torch.cat([target_vec, context_vec], dim=1)
        else:
            combined = target_vec
        
        # Decode to predictions
        logits = self.decoder(combined)
        
        # Empty room prediction (auxiliary task)
        if self.use_empty_room_head and return_empty_room_logits:
            empty_room_logits = self.empty_room_classifier(combined).squeeze(-1)
            return logits, empty_room_logits
        
        return logits


def test_model():
    """Test model with synthetic data."""
    print("Testing DeepSetsModel...")
    
    batch_size = 4
    num_operations = 388
    num_rooms = 11
    max_context_rooms = 5
    feature_dim = num_operations + num_rooms
    
    # Create model
    model = DeepSetsModel(
        num_operations=num_operations,
        num_rooms=num_rooms,
        embedding_dim=128,
        hidden_dim=256,
        pooling="mean",
        use_context=True,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create synthetic inputs
    X = torch.rand(batch_size, feature_dim)  # Target room features
    context = torch.rand(batch_size, max_context_rooms, feature_dim)  # Context rooms
    context_mask = torch.ones(batch_size, max_context_rooms, dtype=torch.bool)
    context_mask[0, 3:] = False  # Room 0 has only 3 context rooms
    context_mask[1, 4:] = False  # Room 1 has 4 context rooms
    
    # Forward pass
    logits = model(X, context, context_mask)
    
    print(f"Input shape: {X.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Logit range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Test without context
    model_no_context = DeepSetsModel(
        num_operations=num_operations,
        num_rooms=num_rooms,
        use_context=False,
    )
    logits_no_context = model_no_context(X)
    print(f"\nWithout context - output shape: {logits_no_context.shape}")
    
    # Test different pooling strategies
    for pooling in ["mean", "sum", "max"]:
        model_pool = DeepSetsModel(pooling=pooling, use_context=True)
        logits_pool = model_pool(X, context, context_mask)
        print(f"Pooling={pooling}: output range [{logits_pool.min():.3f}, {logits_pool.max():.3f}]")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_model()
