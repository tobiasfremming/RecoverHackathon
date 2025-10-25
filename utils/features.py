"""
Feature Engineering Utilities

Functions for processing and transforming features for the Deep Sets model.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


def normalize_numeric_features(
    values: torch.Tensor,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, float, float]:
    """
    Normalize numeric features to zero mean and unit variance.
    
    Args:
        values: Tensor to normalize
        mean: Pre-computed mean (if None, compute from values)
        std: Pre-computed std (if None, compute from values)
        eps: Small constant for numerical stability
    
    Returns:
        normalized: Normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization
    """
    if mean is None:
        mean = values.mean().item()
    if std is None:
        std = values.std().item()
    
    normalized = (values - mean) / (std + eps)
    return normalized, mean, std


def cyclical_encode_month(month: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode month as sin/cos components to capture cyclical nature.
    
    Args:
        month: Tensor of month values (1-12)
    
    Returns:
        month_sin: Sine component
        month_cos: Cosine component
    """
    # Convert month to radians (0 to 2*pi)
    month_rad = (month / 12.0) * 2 * np.pi
    month_sin = torch.sin(month_rad)
    month_cos = torch.cos(month_rad)
    return month_sin, month_cos


def compute_class_weights(
    tickets_path: str = "data/hackathon-recover-x-cogito/tickets.csv",
    num_clusters: int = 388,
    weight_type: str = "inverse_freq",
    smoothing: float = 0.1
) -> torch.Tensor:
    """
    Compute class weights based on operation frequency from tickets.csv.
    
    Args:
        tickets_path: Path to tickets.csv file
        num_clusters: Number of operation clusters
        weight_type: Type of weighting ('inverse_freq', 'sqrt_inverse_freq', 'balanced')
        smoothing: Smoothing factor to prevent extreme weights
    
    Returns:
        weights: Tensor of shape (num_clusters,) with class weights
    """
    import pandas as pd
    
    # Load tickets data
    tickets = pd.read_csv(tickets_path)
    
    # Create frequency array
    frequencies = np.zeros(num_clusters)
    for _, row in tickets.iterrows():
        code = int(row['work_operation_cluster_code'])
        count = int(row['n_tickets'])
        frequencies[code] = count
    
    # Avoid division by zero
    frequencies = frequencies + smoothing
    
    # Compute weights based on type
    if weight_type == "inverse_freq":
        weights = 1.0 / frequencies
    elif weight_type == "sqrt_inverse_freq":
        weights = 1.0 / np.sqrt(frequencies)
    elif weight_type == "balanced":
        total = frequencies.sum()
        weights = total / (num_clusters * frequencies)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
    
    # Normalize weights
    weights = weights / weights.mean()
    
    return torch.tensor(weights, dtype=torch.float32)


def extract_operation_codes_from_one_hot(one_hot: torch.Tensor) -> List[List[int]]:
    """
    Convert one-hot encoded operations back to operation codes.
    
    Args:
        one_hot: Tensor of shape (batch, num_clusters) with binary values
    
    Returns:
        codes: List of lists, where each inner list contains operation codes
    """
    batch_size = one_hot.shape[0]
    codes = []
    
    for i in range(batch_size):
        sample_codes = torch.where(one_hot[i] > 0.5)[0].tolist()
        codes.append(sample_codes)
    
    return codes


def codes_to_padded_tensor(
    codes_list: List[List[int]],
    max_length: Optional[int] = None,
    pad_value: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert list of operation code lists to padded tensor.
    
    Args:
        codes_list: List of lists of operation codes
        max_length: Maximum length (if None, use max length in batch)
        pad_value: Value to use for padding
    
    Returns:
        codes_tensor: Padded tensor of shape (batch, max_length)
        mask: Boolean mask indicating valid positions
    """
    if max_length is None:
        max_length = max(len(codes) for codes in codes_list)
    
    batch_size = len(codes_list)
    codes_tensor = torch.full((batch_size, max_length), pad_value, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    for i, codes in enumerate(codes_list):
        length = min(len(codes), max_length)
        codes_tensor[i, :length] = torch.tensor(codes[:length], dtype=torch.long)
        mask[i, :length] = True
    
    return codes_tensor, mask


def process_batch_for_model(
    batch: Dict,
    normalize_stats: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, torch.Tensor]:
    """
    Process a batch from DataLoader into model-ready format.
    
    The DataLoader with collate_fn returns:
    - X: (batch, feature_dim) - concatenated [operations, room_cluster]
    - Y: (batch, num_clusters) - target operations one-hot
    - context: (batch, max_rooms, feature_dim) - context rooms
    - context_mask: (batch, max_rooms) - validity mask for context
    
    We need to extract and process these for the model.
    
    Args:
        batch: Batch dictionary from DataLoader
        normalize_stats: Optional normalization statistics
    
    Returns:
        processed: Dictionary with processed tensors ready for model
    """
    # Extract X (operations + room_cluster concatenated)
    X_full = batch["X"]  # (batch, num_clusters + num_rooms)
    num_clusters = 388
    num_rooms = 11
    
    # Split X into operations and room_cluster
    X_operations = X_full[:, :num_clusters]  # (batch, num_clusters)
    room_cluster_one_hot = X_full[:, num_clusters:]  # (batch, num_rooms)
    
    # Convert X operations from one-hot to codes
    X_codes_list = extract_operation_codes_from_one_hot(X_operations)
    X_codes, X_mask = codes_to_padded_tensor(X_codes_list)
    
    # Extract context and mask
    context = batch["context"]
    context_mask = batch["context_mask"]
    
    # Extract Y (targets)
    Y = batch["Y"]
    
    processed = {
        "X_codes": X_codes,
        "X_mask": X_mask,
        "context": context,
        "context_mask": context_mask,
        "room_cluster_one_hot": room_cluster_one_hot,
        "Y": Y
    }
    
    return processed


def add_metadata_to_batch(
    batch: Dict[str, torch.Tensor],
    metadata: Dict[str, torch.Tensor],
    normalize_stats: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, torch.Tensor]:
    """
    Add metadata features to the processed batch.
    
    Args:
        batch: Processed batch from process_batch_for_model
        metadata: Dictionary with metadata tensors
        normalize_stats: Normalization statistics for numeric features
    
    Returns:
        batch_with_metadata: Combined batch with metadata
    """
    # Extract metadata
    insurance_company_one_hot = metadata["insurance_company_one_hot"]
    office_distance = metadata["office_distance"].unsqueeze(-1)  # (batch, 1)
    case_creation_year = metadata["case_creation_year"].unsqueeze(-1)  # (batch, 1)
    case_creation_month = metadata["case_creation_month"].unsqueeze(-1).float()  # (batch, 1)
    
    # Normalize numeric features
    if normalize_stats is not None:
        if "office_distance" in normalize_stats:
            mean, std = normalize_stats["office_distance"]
            office_distance = (office_distance - mean) / (std + 1e-8)
        if "case_creation_year" in normalize_stats:
            mean, std = normalize_stats["case_creation_year"]
            case_creation_year = (case_creation_year - mean) / (std + 1e-8)
    
    # Add metadata to batch
    batch_with_metadata = {
        **batch,
        "insurance_company_one_hot": insurance_company_one_hot,
        "office_distance": office_distance,
        "case_creation_year": case_creation_year,
        "case_creation_month": case_creation_month
    }
    
    return batch_with_metadata


class FeatureNormalizer:
    """
    Stateful feature normalizer that computes and stores normalization statistics.
    """
    
    def __init__(self):
        self.stats = {}
    
    def fit(self, dataset, feature_names: List[str]):
        """
        Compute normalization statistics from dataset.
        
        Args:
            dataset: HackathonDataset or similar
            feature_names: List of feature names to normalize
        """
        # Collect feature values
        feature_values = {name: [] for name in feature_names}
        
        for i in range(len(dataset)):
            sample = dataset[i]
            for name in feature_names:
                if name in sample:
                    feature_values[name].append(sample[name])
        
        # Compute statistics
        for name in feature_names:
            values = torch.tensor(feature_values[name], dtype=torch.float32)
            mean = values.mean().item()
            std = values.std().item()
            self.stats[name] = (mean, std)
    
    def transform(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalize features using stored statistics.
        
        Args:
            features: Dictionary of features to normalize
        
        Returns:
            normalized_features: Normalized features
        """
        normalized = {}
        for name, value in features.items():
            if name in self.stats:
                mean, std = self.stats[name]
                normalized[name] = (value - mean) / (std + 1e-8)
            else:
                normalized[name] = value
        return normalized
    
    def save(self, path: str):
        """Save normalization statistics to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.stats, f)
    
    def load(self, path: str):
        """Load normalization statistics from file."""
        import json
        with open(path, 'r') as f:
            self.stats = json.load(f)


if __name__ == "__main__":
    # Test feature utilities
    print("Testing feature engineering utilities...")
    
    # Test cyclical encoding
    months = torch.tensor([1, 3, 6, 9, 12])
    month_sin, month_cos = cyclical_encode_month(months)
    print(f"\nMonth encoding:")
    print(f"Months: {months.tolist()}")
    print(f"Sin: {month_sin.tolist()}")
    print(f"Cos: {month_cos.tolist()}")
    
    # Test class weights
    try:
        weights = compute_class_weights(smoothing=1.0)
        print(f"\nClass weights computed:")
        print(f"  Min weight: {weights.min():.4f}")
        print(f"  Max weight: {weights.max():.4f}")
        print(f"  Mean weight: {weights.mean():.4f}")
    except Exception as e:
        print(f"\nCould not compute class weights: {e}")
    
    # Test one-hot to codes conversion
    batch_size = 3
    num_clusters = 10
    one_hot = torch.zeros(batch_size, num_clusters)
    one_hot[0, [1, 3, 5]] = 1
    one_hot[1, [2, 7]] = 1
    one_hot[2, [0, 4, 8, 9]] = 1
    
    codes = extract_operation_codes_from_one_hot(one_hot)
    print(f"\nOne-hot to codes conversion:")
    for i, c in enumerate(codes):
        print(f"  Sample {i}: {c}")
    
    # Test codes to padded tensor
    codes_tensor, mask = codes_to_padded_tensor(codes)
    print(f"\nPadded codes tensor:")
    print(f"  Shape: {codes_tensor.shape}")
    print(f"  Codes:\n{codes_tensor}")
    print(f"  Mask:\n{mask}")
    
    print("\nAll tests passed!")
