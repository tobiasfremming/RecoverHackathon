"""
Configuration for Deep Sets model training.

This module contains all hyperparameters and settings in one place.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Architecture
    num_operations: int = 388
    """Number of possible work operations"""
    
    num_rooms: int = 11
    """Number of room types"""
    
    embedding_dim: int = 128
    """Dimension of operation embeddings"""
    
    hidden_dim: int = 256
    """Hidden dimension for MLP layers"""
    
    num_layers: int = 3
    """Number of layers in MLP"""
    
    pooling: str = "mean"
    """Pooling strategy: 'mean', 'sum', or 'max'"""
    
    dropout: float = 0.1
    """Dropout probability"""
    
    use_context: bool = True
    """Whether to use context from other rooms"""
    
    use_metadata: bool = False
    """Whether to use project metadata (insurance company, etc.)"""


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    learning_rate: float = 1e-4
    """Learning rate for optimizer"""
    
    weight_decay: float = 1e-5
    """L2 regularization"""
    
    batch_size: int = 32
    """Batch size for training"""
    
    num_workers: int = 0
    """Number of data loading workers (0 for Windows compatibility)"""
    
    num_epochs: int = 2
    """Number of training epochs"""
    
    gradient_clip: float = 1.0
    """Maximum gradient norm"""
    
    # Learning rate schedule
    use_scheduler: bool = True
    """Whether to use learning rate scheduler"""
    
    scheduler_patience: int = 5
    """Patience for ReduceLROnPlateau"""
    
    scheduler_factor: float = 0.5
    """Factor to reduce LR by"""
    
    # Properties for backward compatibility
    @property
    def lr(self) -> float:
        """Alias for learning_rate"""
        return self.learning_rate
    
    # Early stopping
    early_stopping_patience: int = 10
    """Stop if no improvement for N epochs"""
    
    # Checkpointing
    save_every: int = 5
    """Save checkpoint every N epochs"""
    
    checkpoint_dir: str = "deepset/checkpoints"
    """Directory to save checkpoints"""


@dataclass
class LossConfig:
    """Loss function configuration."""
    
    # Focal loss parameters
    use_focal_loss: bool = True
    """Use focal loss instead of BCE"""
    
    focal_alpha: float = 0.75
    """Weight for positive class (0.5 = balanced, >0.5 = focus on positives)"""
    
    focal_gamma: float = 2.0
    """Focusing parameter (0 = BCE, >0 = down-weight easy examples)"""
    
    # Class weighting
    use_class_weights: bool = True
    """Weight operations by inverse frequency"""
    
    weight_scale: float = 1.0
    """Scale factor for class weights"""
    
    # Multi-task learning
    use_auxiliary_loss: bool = False
    """Predict if room is complete (no missing operations)"""
    
    auxiliary_weight: float = 0.1
    """Weight for auxiliary loss"""
    
    # Properties for backward compatibility
    @property
    def loss_type(self) -> str:
        """Return 'focal' or 'bce' based on use_focal_loss"""
        return "focal" if self.use_focal_loss else "bce"
    
    @property
    def pos_weight_scale(self) -> float:
        """Alias for weight_scale"""
        return self.weight_scale


@dataclass
class DataConfig:
    """Data configuration."""
    
    data_root: str = "data"
    """Root directory for data"""
    
    download: bool = False
    """Download data if not present"""
    
    seed: int = 42
    """Random seed for reproducibility"""
    
    num_workers: int = 0
    """Number of data loading workers (0 for Windows compatibility)"""
    
    # Sampling strategy
    sampling_strategy: Optional[list[dict]] = None
    """Custom sampling strategy (None = use default)"""
    
    def __post_init__(self):
        if self.sampling_strategy is None:
            self.sampling_strategy = [
                {
                    "subset_size": 0.7,
                    "sample_pct": 0.5,
                    "use_balanced_data": True,
                    "use_sampled_calculus": True,
                },
                {
                    "subset_size": 0.3,
                    "sample_pct": 0.3,
                    "use_balanced_data": False,
                    "use_sampled_calculus": True,
                },
            ]


@dataclass
class Config:
    """Complete configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment tracking
    experiment_name: str = "deepset_baseline"
    """Name for this experiment"""
    
    log_interval: int = 100
    """Log every N batches"""
    
    device: str = "cuda"
    """Device to use (cuda/cpu)"""


def get_config(preset: str = "default") -> Config:
    """Get configuration by preset name."""
    
    if preset == "default":
        return Config()
    
    elif preset == "fast":
        # For quick experimentation
        config = Config()
        config.training.num_epochs = 10
        config.training.batch_size = 64
        config.training.early_stopping_patience = 5
        config.experiment_name = "deepset_fast"
        return config
    
    elif preset == "strong":
        # For best performance
        config = Config()
        config.model.embedding_dim = 256
        config.model.hidden_dim = 512
        config.model.dropout = 0.2
        config.training.num_epochs = 100
        config.training.batch_size = 16
        config.training.learning_rate = 5e-5
        config.loss.focal_alpha = 0.85
        config.experiment_name = "deepset_strong"
        return config
    
    elif preset == "debug":
        # For debugging
        config = Config()
        config.training.num_epochs = 2
        config.training.batch_size = 8
        config.data.num_workers = 0
        config.experiment_name = "deepset_debug"
        return config
    
    elif preset == "optimized":
        # Optimized for competition scoring (FN penalty 2x FP penalty)
        config = Config()
        # Model: Moderate size for good performance
        config.model.embedding_dim = 192
        config.model.hidden_dim = 384
        config.model.dropout = 0.15
        # Training: Longer training, moderate batch size
        config.training.num_epochs = 50
        config.training.batch_size = 32
        config.training.learning_rate = 8e-5
        config.training.early_stopping_patience = 15
        # Loss: Reduce focal_alpha to decrease FP rate
        # Lower alpha = less focus on positives = fewer FPs
        config.loss.focal_alpha = 0.65  # Down from 0.75
        config.loss.focal_gamma = 2.5    # Up from 2.0 to focus on hard examples
        config.experiment_name = "deepset_optimized"
        return config
    
    elif preset == "competition":
        # Use competition-aligned loss function
        config = Config()
        # Model: Same as optimized
        config.model.embedding_dim = 192
        config.model.hidden_dim = 384
        config.model.dropout = 0.15
        # Training: Longer training
        config.training.num_epochs = 50
        config.training.batch_size = 32
        config.training.learning_rate = 8e-5
        config.training.early_stopping_patience = 15
        # Loss: Use competition-aligned loss (weights FN 2x more than FP)
        config.loss.use_focal_loss = False  # Disable standard focal loss
        config.loss.focal_alpha = 0.65
        config.loss.focal_gamma = 2.5
        config.experiment_name = "deepset_competition"
        return config
    
    elif preset == "aggressive":
        # Most aggressive FP reduction
        config = Config()
        # Model: Larger for better calibration
        config.model.embedding_dim = 256
        config.model.hidden_dim = 512
        config.model.dropout = 0.2
        # Training: Very long training
        config.training.num_epochs = 75
        config.training.batch_size = 24
        config.training.learning_rate = 5e-5
        config.training.early_stopping_patience = 20
        # Loss: Very conservative settings
        config.loss.focal_alpha = 0.60  # Even lower alpha
        config.loss.focal_gamma = 3.0   # Focus heavily on hard examples
        config.experiment_name = "deepset_aggressive"
        return config
    
    elif preset == "ultimate":
        # Ultimate optimization: All features enabled
        config = Config()
        # Model: Large with auxiliary task
        config.model.embedding_dim = 256
        config.model.hidden_dim = 512
        config.model.dropout = 0.18
        # Training: Long training with moderate batch
        config.training.num_epochs = 60
        config.training.batch_size = 28
        config.training.learning_rate = 6e-5
        config.training.early_stopping_patience = 18
        # Loss: Competition-aligned focal + auxiliary task
        config.loss.use_focal_loss = False  # Will use competition_focal
        config.loss.focal_alpha = 0.62
        config.loss.focal_gamma = 2.8
        config.loss.use_auxiliary_loss = True  # Enable empty room prediction
        config.loss.auxiliary_weight = 0.15    # Weight for auxiliary task
        config.experiment_name = "deepset_ultimate"
        return config
    
    else:
        raise ValueError(f"Unknown preset: {preset}")


if __name__ == "__main__":
    # Test configurations
    for preset in ["default", "fast", "strong", "debug"]:
        cfg = get_config(preset)
        print(f"\n{preset.upper()} Config:")
        print(f"  Model: {cfg.model.embedding_dim}d embeddings, {cfg.model.hidden_dim}d hidden")
        print(f"  Training: {cfg.training.num_epochs} epochs, batch={cfg.training.batch_size}, lr={cfg.training.learning_rate}")
        print(f"  Loss: focal_alpha={cfg.loss.focal_alpha}, gamma={cfg.loss.focal_gamma}")
