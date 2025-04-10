import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --- Project Root ---
# Assumes config.py is located at the project root or a standard location
# Adjust if necessary
PROJECT_ROOT = Path(__file__).parent.resolve()

# --- Path Configurations ---
@dataclass
class PathsConfig:
    """Configuration for project paths."""
    root: Path = PROJECT_ROOT
    data: Path = root / "data"
    raw_data: Path = data / "raw"
    processed_data: Path = data / "processed"
    models: Path = root / "models"
    results: Path = root / "results"
    logs: Path = root / "logs"
    checkpoints: Path = models / "checkpoints"

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data.mkdir(parents=True, exist_ok=True)
        self.raw_data.mkdir(parents=True, exist_ok=True)
        self.processed_data.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.checkpoints.mkdir(parents=True, exist_ok=True)

# --- Model Parameters ---
@dataclass
class ModelConfig:
    """Configuration for the forecasting model."""
    model_type: str = "LSTM"  # Example: LSTM, GRU, Transformer
    input_sequence_length: int = 30  # Days of past data to use
    output_sequence_length: int = 7   # Days to forecast
    num_features: int = 5  # Example: sentiment, engagement, volume, price, related_trends
    hidden_layer_size: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.2
    embedding_dim: Optional[int] = None # For models using embeddings
    attention_heads: Optional[int] = None # For Transformer models

# --- Training Parameters ---
@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: str = "Adam"  # Example: Adam, SGD, RMSprop
    loss_function: str = "MSE" # Example: MSE, MAE
    validation_split: float = 0.2 # Percentage of data for validation
    early_stopping_patience: int = 10
    gradient_clipping_value: Optional[float] = 1.0
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    seed: int = 42

# --- Environment Configuration ---
@dataclass
class EnvironmentConfig:
    """Configuration for the execution environment."""
    device: str = "cuda" if os.environ.get("USE_CUDA", "true").lower() == "true" else "cpu" # Or 'mps' for Apple Silicon
    log_level: int = logging.INFO
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file_name: str = "chronomeme_forecaster.log"
    wandb_project: Optional[str] = "ChronoMemeForecaster" # Weights & Biases project name
    wandb_entity: Optional[str] = None # Weights & Biases entity (username or team)
    enable_wandb: bool = True # Toggle W&B logging

# --- Instantiate Configurations ---
paths_config = PathsConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
env_config = EnvironmentConfig()

# --- Setup Logging ---
log_file_path = paths_config.logs / env_config.log_file_name
logging.basicConfig(
    level=env_config.log_level,
    format=env_config.log_format,
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Example Usage (can be removed in final version) ---
if __name__ == "__main__":
    logger.info("Configuration loaded successfully.")
    logger.info(f"Project Root: {paths_config.root}")
    logger.info(f"Data Path: {paths_config.data}")
    logger.info(f"Model Type: {model_config.model_type}")
    logger.info(f"Input Sequence Length: {model_config.input_sequence_length}")
    logger.info(f"Epochs: {training_config.epochs}")
    logger.info(f"Batch Size: {training_config.batch_size}")
    logger.info(f"Device: {env_config.device}")
    logger.info(f"Log Level: {logging.getLevelName(env_config.log_level)}")
    logger.info(f"W&B Enabled: {env_config.enable_wandb}")