"""
Utility functions for the phage-set-transformer package.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def create_output_directory(base_dir: Optional[str] = None, prefix: str = "pst_results", timestamp: Optional[str] = None) -> str:
    """
    Create output directory with timestamp.
    
    Args:
        base_dir: Base directory to create output in (if None, uses current directory)
        prefix: Prefix for the output directory name
        timestamp: Optional timestamp string (if None, generates current timestamp)
        
    Returns:
        Path to created directory
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if base_dir:
        output_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    else:
        output_dir = f"{prefix}_{timestamp}"
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    
    return output_dir

def get_device() -> torch.device:
    """
    Get the appropriate device for computation.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def save_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save config in
    """
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4, cls=NumpyEncoder)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Saved configuration to {config_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


class EarlyStopping:
    """Early stopping to stop training when a metric has stopped improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' or 'min' for metric optimization direction
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, current_metric: float, model: Optional[torch.nn.Module] = None) -> float:
        """
        Check if early stopping should trigger.
        
        Args:
            current_metric: Current metric value
            model: Optional model to save state
            
        Returns:
            Best metric value so far
        """
        if self.best_metric is None:
            self.best_metric = current_metric
            if model is not None:
                self.best_model_state = model.state_dict()
        else:
            # For 'max' mode, improvement means current_metric >= best_metric + min_delta
            if self.mode == 'max':
                if current_metric < (self.best_metric + self.min_delta):
                    self.counter += 1
                else:
                    self.best_metric = current_metric
                    self.counter = 0
                    if model is not None:
                        self.best_model_state = model.state_dict()
            else:  # 'min' mode
                if current_metric > (self.best_metric - self.min_delta):
                    self.counter += 1
                else:
                    self.best_metric = current_metric
                    self.counter = 0
                    if model is not None:
                        self.best_model_state = model.state_dict()

            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.best_metric


def get_scheduler(scheduler_type: str, 
                  optimizer: torch.optim.Optimizer, 
                  num_train_steps: int, 
                  warmup_ratio: float = 0.1) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler.
    
    Args:
        scheduler_type: Type of scheduler to create
        optimizer: Optimizer to schedule
        num_train_steps: Total number of training steps
        warmup_ratio: Ratio of steps for warmup
        
    Returns:
        Learning rate scheduler or None
    """
    if scheduler_type == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=num_train_steps,
            pct_start=warmup_ratio
        )
    elif scheduler_type == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_train_steps
        )
    elif scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=3,
            factor=0.5
        )
    elif scheduler_type == "linear_warmup_decay":
        # Create a simple linear warmup followed by linear decay
        def lr_lambda(current_step: int) -> float:
            warmup_steps = int(num_train_steps * warmup_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(num_train_steps - current_step) / float(max(1, num_train_steps - warmup_steps))
            )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:  # "none" or any other value
        return None
