"""
phage-set-transformer: Set Transformer architecture for phage-host interaction prediction.

This package provides tools for training and optimizing set transformer models
to predict interactions between bacterial strains and phages using gene embeddings.
"""

from .optimization import run_optimization
from .training import train_model_with_params
from .evaluation import predict, load_model

__version__ = "0.1.0"
__all__ = ["run_optimization", "train_model_with_params", "predict", "load_model"]
