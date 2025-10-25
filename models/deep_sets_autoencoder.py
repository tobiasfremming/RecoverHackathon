"""
Deep Sets Denoising Set Autoencoder for Work Operations Prediction

This model implements a permutation-invariant set autoencoder based on the Deep Sets architecture.
It learns to predict masked work operations given:
- Visible operations in a room (variable-length set)
- Context operations from other rooms in the project (calculus)
- Project metadata (insurance company, location, dates)
- Room type (cluster)

Architecture:
1. Operation Embedding: Maps operation codes to dense vectors
2. Set Encoder: Aggregates operation embeddings via permutation-invariant pooling
3. Context Encoder: Aggregates calculus room information with attention
4. Metadata Encoder: Processes categorical and numeric metadata features
5. Decoder MLP: Predicts missing operations with sigmoid outputs
6. Auxiliary Head: Predicts if room is complete (no missing operations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SetPooling(nn.Module):
    """Permutation-invariant pooling for sets."""
    
    def __init__(self, pooling_type: str = "mean"):
        super().__init__()
        assert pooling_type in ["mean", "sum", "max"], f"Invalid pooling type: {pooling_type}"
        self.pooling_type = pooling_type
    
    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, set_size, embedding_dim)
            mask: (batch_size, set_size) - True for valid elements
        Returns:
            pooled: (batch_size, embedding_dim)
        """
        if mask is not None:
            # Expand mask for broadcasting
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, set_size, 1)
            embeddings = embeddings * mask_expanded
        
        if self.pooling_type == "mean":
            if mask is not None:
                # Compute mean only over valid elements
                sum_embeddings = embeddings.sum(dim=1)
                count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
                return sum_embeddings / count
            else:
                return embeddings.mean(dim=1)
        elif self.pooling_type == "sum":
            return embeddings.sum(dim=1)
        elif self.pooling_type == "max":
            if mask is not None:
                # Set masked positions to very negative values
                embeddings = embeddings.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            return embeddings.max(dim=1)[0]


class AttentivePooling(nn.Module):
    """Attention-based pooling for context rooms."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize attention weights
        for module in self.attention.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, num_rooms, embedding_dim)
            mask: (batch_size, num_rooms) - True for valid rooms
        Returns:
            pooled: (batch_size, embedding_dim)
        """
        # Compute attention scores
        scores = self.attention(embeddings).squeeze(-1)  # (batch, num_rooms)
        
        # Clamp scores to prevent extreme values before masking
        scores = torch.clamp(scores, min=-10.0, max=10.0)
        
        if mask is not None:
            # Mask out invalid rooms
            scores = scores.masked_fill(~mask, -1e9)  # Use -1e9 instead of -inf for numerical stability
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, num_rooms, 1)
        
        # Check for NaN in weights
        if torch.isnan(weights).any():
            # Fallback to uniform weighting
            if mask is not None:
                uniform_weights = mask.float() / mask.sum(dim=1, keepdim=True).clamp(min=1)
                weights = uniform_weights.unsqueeze(-1)
            else:
                weights = torch.ones_like(weights) / weights.shape[1]
        
        # Weighted sum
        pooled = (embeddings * weights).sum(dim=1)  # (batch, embedding_dim)
        
        return pooled


class MetadataEncoder(nn.Module):
    """Encodes project metadata features."""
    
    def __init__(
        self,
        num_companies: int = 14,
        num_rooms: int = 11,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Calculate total input dimension
        # One-hot: insurance_company (14) + room_cluster (11) = 25
        # Numeric: office_distance (1) + year (1) + month_sin (1) + month_cos (1) = 4
        # Total = 29
        input_dim = num_companies + num_rooms + 4
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        insurance_company_one_hot: torch.Tensor,
        room_cluster_one_hot: torch.Tensor,
        office_distance: torch.Tensor,
        case_creation_year: torch.Tensor,
        case_creation_month: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            insurance_company_one_hot: (batch, num_companies)
            room_cluster_one_hot: (batch, num_rooms)
            office_distance: (batch, 1) - normalized
            case_creation_year: (batch, 1) - normalized
            case_creation_month: (batch, 1) - raw month (1-12)
        Returns:
            encoded: (batch, hidden_dim)
        """
        # Cyclical encoding for month
        month_rad = (case_creation_month / 12.0) * 2 * 3.14159265359
        month_sin = torch.sin(month_rad)
        month_cos = torch.cos(month_rad)
        
        # Concatenate all features
        features = torch.cat([
            insurance_company_one_hot,
            room_cluster_one_hot,
            office_distance,
            case_creation_year,
            month_sin,
            month_cos
        ], dim=-1)
        
        return self.encoder(features)


class DeepSetsAutoencoder(nn.Module):
    """
    Deep Sets Denoising Set Autoencoder for predicting masked work operations.
    
    Args:
        num_clusters: Number of work operation clusters (388)
        embedding_dim: Dimension of operation embeddings
        hidden_dim: Hidden dimension for MLPs
        num_companies: Number of insurance companies (14)
        num_rooms: Number of room clusters (11)
        pooling_type: Type of set pooling ("mean", "sum", "max")
        use_attention: Whether to use attention for context pooling
        dropout: Dropout probability
        use_auxiliary_head: Whether to predict "room complete" signal
    """
    
    def __init__(
        self,
        num_clusters: int = 388,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_companies: int = 14,
        num_rooms: int = 11,
        pooling_type: str = "mean",
        use_attention: bool = True,
        dropout: float = 0.2,
        use_auxiliary_head: bool = True
    ):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.use_auxiliary_head = use_auxiliary_head
        
        # Operation embedding layer with proper initialization
        self.operation_embedding = nn.Embedding(num_clusters, embedding_dim)
        nn.init.xavier_uniform_(self.operation_embedding.weight)
        
        # Set pooling for target room operations
        self.target_pooling = SetPooling(pooling_type)
        
        # Context projection - initialize here to avoid dynamic creation
        self.context_projection = nn.Linear(num_clusters + num_rooms, embedding_dim)
        nn.init.xavier_uniform_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)
        
        # Context pooling (for calculus rooms)
        if use_attention:
            # Context features include operations + room cluster one-hot
            # After pooling, each room is represented by embedding_dim
            self.context_pooling = AttentivePooling(embedding_dim, hidden_dim=128)
        else:
            self.context_pooling = SetPooling(pooling_type)
        
        # Metadata encoder
        self.metadata_encoder = MetadataEncoder(
            num_companies=num_companies,
            num_rooms=num_rooms,
            hidden_dim=64,
            dropout=dropout
        )
        
        # Decoder MLP with layer normalization for stability
        # Input: target_pooled (embedding_dim) + context_pooled (embedding_dim) + metadata (64)
        decoder_input_dim = embedding_dim + embedding_dim + 64
        
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_clusters)
        )
        
        # Initialize decoder weights
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Auxiliary head for "room complete" prediction
        if use_auxiliary_head:
            self.complete_head = nn.Sequential(
                nn.Linear(decoder_input_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            # Initialize auxiliary head weights
            for module in self.complete_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def encode_operations(self, operation_codes: torch.Tensor) -> torch.Tensor:
        """
        Embed operation codes.
        Args:
            operation_codes: (batch, set_size) - operation cluster codes
        Returns:
            embeddings: (batch, set_size, embedding_dim)
        """
        return self.operation_embedding(operation_codes)
    
    def encode_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode context from other rooms (calculus).
        
        Args:
            context: (batch, max_rooms, feature_dim) - concatenated operations + room_cluster
            context_mask: (batch, max_rooms) - True for valid rooms
        Returns:
            context_encoded: (batch, embedding_dim)
        """
        # Ensure context is float type
        context = context.float()
        
        # Replace NaN and Inf in context with zeros
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clamp context values to reasonable range
        context = torch.clamp(context, min=-10.0, max=10.0)
        
        # Project context features to embedding dimension
        context_projected = self.context_projection(context)  # (batch, max_rooms, embedding_dim)
        
        # Pool across rooms
        context_pooled = self.context_pooling(context_projected, context_mask)
        
        return context_pooled
    
    def forward(
        self,
        X_codes: torch.Tensor,
        X_mask: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        insurance_company_one_hot: torch.Tensor,
        room_cluster_one_hot: torch.Tensor,
        office_distance: torch.Tensor,
        case_creation_year: torch.Tensor,
        case_creation_month: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            X_codes: (batch, max_ops) - visible operation codes in target room
            X_mask: (batch, max_ops) - True for valid operations
            context: (batch, max_rooms, feature_dim) - context from other rooms
            context_mask: (batch, max_rooms) - True for valid rooms
            insurance_company_one_hot: (batch, num_companies)
            room_cluster_one_hot: (batch, num_rooms)
            office_distance: (batch, 1)
            case_creation_year: (batch, 1)
            case_creation_month: (batch, 1)
        
        Returns:
            logits: (batch, num_clusters) - raw logits for each operation (before sigmoid)
            complete_logit: (batch, 1) - raw logit for room completeness (before sigmoid), optional
        """
        # Encode target room operations
        X_embeddings = self.encode_operations(X_codes)  # (batch, max_ops, embedding_dim)
        X_pooled = self.target_pooling(X_embeddings, X_mask)  # (batch, embedding_dim)
        
        # Encode context rooms
        context_pooled = self.encode_context(context, context_mask)  # (batch, embedding_dim)
        
        # Encode metadata
        metadata_encoded = self.metadata_encoder(
            insurance_company_one_hot,
            room_cluster_one_hot,
            office_distance,
            case_creation_year,
            case_creation_month
        )  # (batch, 64)
        
        # Concatenate all representations
        combined = torch.cat([X_pooled, context_pooled, metadata_encoded], dim=-1)
        
        # Clamp combined features to prevent extreme values
        combined = torch.clamp(combined, min=-10.0, max=10.0)
        
        # Decode to predict missing operations
        logits = self.decoder(combined)  # (batch, num_clusters)
        
        # Clamp logits to prevent extreme values
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        
        # Auxiliary head for "room complete" prediction
        complete_logit = None
        if self.use_auxiliary_head:
            complete_logit = self.complete_head(combined)
            complete_logit = torch.clamp(complete_logit, min=-20.0, max=20.0)
        
        return logits, complete_logit


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    num_clusters = 388
    num_companies = 14
    num_rooms = 11
    max_ops = 10
    max_context_rooms = 5
    feature_dim = num_clusters + num_rooms  # context feature dimension
    
    model = DeepSetsAutoencoder(
        num_clusters=num_clusters,
        embedding_dim=128,
        hidden_dim=256,
        num_companies=num_companies,
        num_rooms=num_rooms,
        pooling_type="mean",
        use_attention=True,
        dropout=0.2,
        use_auxiliary_head=True
    )
    
    # Create dummy inputs
    X_codes = torch.randint(0, num_clusters, (batch_size, max_ops))
    X_mask = torch.ones(batch_size, max_ops, dtype=torch.bool)
    X_mask[0, 5:] = False  # First sample has only 5 operations
    
    context = torch.randn(batch_size, max_context_rooms, feature_dim)
    context_mask = torch.ones(batch_size, max_context_rooms, dtype=torch.bool)
    context_mask[1, 3:] = False  # Second sample has only 3 context rooms
    
    insurance_company_one_hot = torch.zeros(batch_size, num_companies)
    insurance_company_one_hot[:, 0] = 1  # All from company 0
    
    room_cluster_one_hot = torch.zeros(batch_size, num_rooms)
    room_cluster_one_hot[:, 2] = 1  # All from room cluster 2
    
    office_distance = torch.rand(batch_size, 1)
    case_creation_year = torch.rand(batch_size, 1)
    case_creation_month = torch.randint(1, 13, (batch_size, 1)).float()
    
    # Forward pass
    predictions, complete_pred = model(
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
    
    print(f"Predictions shape: {predictions.shape}")  # (batch_size, num_clusters)
    print(f"Complete prediction shape: {complete_pred.shape}")  # (batch_size, 1)
    print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Complete prediction range: [{complete_pred.min():.4f}, {complete_pred.max():.4f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
