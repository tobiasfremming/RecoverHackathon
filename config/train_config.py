"""
Training Configuration for Deep Sets Autoencoder

This file contains all hyperparameters and settings for training the model.
Adjust these values to experiment with different configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    
    num_clusters: int = 388
    """Number of work operation clusters"""
    
    embedding_dim: int = 128
    """Dimension of operation embeddings"""
    
    hidden_dim: int = 256
    """Hidden dimension for decoder MLP"""
    
    num_companies: int = 14
    """Number of insurance companies"""
    
    num_rooms: int = 11
    """Number of room clusters"""
    
    pooling_type: str = "mean"
    """Set pooling type: 'mean', 'sum', or 'max'"""
    
    use_attention: bool = True
    """Use attention-based pooling for context rooms"""
    
    dropout: float = 0.2
    """Dropout probability"""
    
    use_auxiliary_head: bool = True
    """Include auxiliary head for 'room complete' prediction"""


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    batch_size: int = 32
    """Batch size for training"""
    
    num_epochs: int = 50
    """Number of training epochs"""
    
    learning_rate: float = 1e-4
    """Initial learning rate"""
    
    weight_decay: float = 1e-4
    """Weight decay for AdamW optimizer"""
    
    grad_clip: float = 0.5
    """Gradient clipping threshold"""
    
    scheduler_type: str = "cosine"
    """LR scheduler: 'cosine', 'plateau', or 'step'"""
    
    scheduler_params: Dict = field(default_factory=lambda: {
        "T_max": 50,  # For cosine annealing
        "eta_min": 1e-6,
        "patience": 5,  # For ReduceLROnPlateau
        "factor": 0.5,
        "step_size": 10,  # For StepLR
        "gamma": 0.1
    })
    """Parameters for learning rate scheduler"""
    
    warmup_epochs: int = 5
    """Number of warmup epochs for learning rate"""
    
    early_stopping_patience: int = 10
    """Patience for early stopping (epochs)"""


@dataclass
class LossConfig:
    """Loss function configuration."""
    
    use_focal_loss: bool = True
    """Use focal loss instead of standard BCE"""
    
    focal_alpha: float = 0.90
    """Focal loss alpha (class balancing) - higher means more weight on positive class"""
    
    focal_gamma: float = 2.0
    """Focal loss gamma (focusing parameter)"""
    
    use_class_weights: bool = True
    """Use class weights from tickets.csv for BCE loss"""
    
    pos_weight_scale: float = 3.0
    """Scale factor for positive class weights"""
    
    main_loss_weight: float = 10.0
    """Weight for main task (operation prediction)"""
    
    auxiliary_loss_weight: float = 0.05
    """Weight for auxiliary 'room complete' loss"""
    
    bce_reduction: str = "mean"
    """BCE loss reduction: 'mean', 'sum', or 'none'"""


@dataclass
class SamplingConfig:
    """Data sampling strategy configuration."""
    
    train_sampling_strategies: List[Dict] = field(default_factory=lambda: [
        {
            "subset_size": 0.5,
            "sample_pct": 0.5,
            "use_balanced_data": True,
            "use_sampled_calculus": True,
        },
        {
            "subset_size": 0.5,
            "sample_pct": 0.3,
            "use_balanced_data": True,
            "use_sampled_calculus": True,
        },
    ])
    """Sampling strategies for training data"""
    
    val_sampling_strategies: List[Dict] = field(default_factory=lambda: [
        {
            "subset_size": 1.0,
            "sample_pct": 0.5,
            "use_balanced_data": False,
            "use_sampled_calculus": False,
        }
    ])
    """Sampling strategies for validation data"""
    
    reshuffle_each_epoch: bool = True
    """Reshuffle training data each epoch"""
    
    num_workers: int = 4
    """Number of workers for DataLoader"""
    
    pin_memory: bool = True
    """Pin memory for faster GPU transfer"""


@dataclass
class InferenceConfig:
    """Inference and threshold tuning configuration."""
    
    default_threshold: float = 0.15
    """Default threshold for binary predictions"""
    
    tune_threshold: bool = True
    """Tune threshold on validation set"""
    
    threshold_search_range: tuple = (0.05, 0.5)
    """Range for threshold search"""
    
    threshold_search_steps: int = 41
    """Number of steps for threshold search"""
    
    per_operation_threshold: bool = False
    """Use different thresholds for each operation"""
    
    use_top_k: bool = False
    """Use top-k predictions instead of thresholding"""
    
    top_k: int = 5
    """Number of top predictions to select (if use_top_k=True)"""


@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    
    seed: int = 42
    """Random seed for reproducibility"""
    
    device: str = "cuda"
    """Device: 'cuda' or 'cpu'"""
    
    data_root: str = "data"
    """Root directory for data"""
    
    checkpoint_dir: str = "checkpoints"
    """Directory to save model checkpoints"""
    
    log_dir: str = "logs"
    """Directory for training logs"""
    
    save_best_only: bool = True
    """Save only the best model checkpoint"""
    
    save_interval: int = 5
    """Save checkpoint every N epochs"""
    
    log_interval: int = 100
    """Log training metrics every N batches"""
    
    validate_interval: int = 1
    """Run validation every N epochs"""
    
    use_wandb: bool = False
    """Use Weights & Biases for logging"""
    
    wandb_project: str = "recover-hackathon"
    """W&B project name"""
    
    wandb_entity: Optional[str] = None
    """W&B entity (username/team)"""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def get_fast_experiment_config() -> ExperimentConfig:
    """Get configuration for fast experimentation."""
    config = ExperimentConfig()
    config.training.num_epochs = 10
    config.training.batch_size = 128
    config.training.learning_rate = 5e-5  # Even lower for stability
    config.training.grad_clip = 0.25  # Tighter gradient clipping
    config.model.dropout = 0.05  # Lower dropout
    config.sampling.num_workers = 2
    return config


def get_production_config() -> ExperimentConfig:
    """Get configuration for production training."""
    config = ExperimentConfig()
    config.model.embedding_dim = 256
    config.model.hidden_dim = 512
    config.training.num_epochs = 100
    config.training.batch_size = 32
    config.training.early_stopping_patience = 15
    config.loss.use_focal_loss = True
    return config


if __name__ == "__main__":
    # Print default configuration
    config = get_default_config()
    print("Default Configuration:")
    print(f"\nModel Config:")
    print(f"  - Embedding dim: {config.model.embedding_dim}")
    print(f"  - Hidden dim: {config.model.hidden_dim}")
    print(f"  - Pooling type: {config.model.pooling_type}")
    print(f"  - Use attention: {config.model.use_attention}")
    
    print(f"\nTraining Config:")
    print(f"  - Batch size: {config.training.batch_size}")
    print(f"  - Epochs: {config.training.num_epochs}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Scheduler: {config.training.scheduler_type}")
    
    print(f"\nLoss Config:")
    print(f"  - Use focal loss: {config.loss.use_focal_loss}")
    print(f"  - Use class weights: {config.loss.use_class_weights}")
    print(f"  - Aux loss weight: {config.loss.auxiliary_loss_weight}")
    
    print(f"\nInference Config:")
    print(f"  - Default threshold: {config.inference.default_threshold}")
    print(f"  - Tune threshold: {config.inference.tune_threshold}")
