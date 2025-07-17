"""
Training functions for the phage-set-transformer package.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
import optuna

from .data import (
    load_embeddings_flexible, 
    filter_interactions_by_strain,
    create_data_loaders
)
from .evaluation import evaluate_full
from .models import FlexibleStrainPhageTransformer, init_attention_weights
from .utils import (
    EarlyStopping, 
    get_scheduler, 
    create_output_directory,
    save_config,
    get_device,
    setup_logging
)
from .visualization import plot_training_history

logger = logging.getLogger(__name__)


def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device, 
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
                use_phage_weights: bool = True) -> Tuple[float, float]:
    """
    Train for one epoch with optional phage-specific weights.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        scheduler: Optional learning rate scheduler
        use_phage_weights: Whether to use phage-specific weights
        
    Returns:
        Tuple of (epoch_loss, epoch_mcc)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in train_loader:
        # Unpack with weights (first 6 elements)
        strain_emb, phage_emb, strain_mask, phage_mask, labels, weights = batch[:6]

        strain_emb = strain_emb.to(device)
        phage_emb = phage_emb.to(device)
        strain_mask = strain_mask.to(device)
        phage_mask = phage_mask.to(device)
        labels = labels.to(device)
        weights = weights.to(device) if use_phage_weights else None

        optimizer.zero_grad()
        logits = model(strain_emb, phage_emb, strain_mask, phage_mask)

        # Use batch-specific weights in the loss calculation if enabled
        if use_phage_weights:
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            criterion = nn.BCEWithLogitsLoss()

        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        total_loss += loss.item()
        preds = torch.sigmoid(logits).reshape(-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(train_loader)
    binary_preds = (np.array(all_preds) > 0.5).astype(int)
    binary_labels = (np.array(all_labels) > 0.5).astype(int)
    epoch_mcc = matthews_corrcoef(binary_labels, binary_preds)

    return epoch_loss, epoch_mcc


def validate(model: nn.Module, 
             val_loader: DataLoader, 
             device: torch.device, 
             use_phage_weights: bool = True) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Validate the model with optional phage-specific weights.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device to run on
        use_phage_weights: Whether to use phage-specific weights
        
    Returns:
        Tuple of (val_loss, val_mcc, binary_preds, binary_labels)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            # Unpack with weights (first 6 elements)
            strain_emb, phage_emb, strain_mask, phage_mask, labels, weights = batch[:6]

            strain_emb = strain_emb.to(device)
            phage_emb = phage_emb.to(device)
            strain_mask = strain_mask.to(device)
            phage_mask = phage_mask.to(device)
            labels = labels.to(device)
            weights = weights.to(device) if use_phage_weights else None

            logits = model(strain_emb, phage_emb, strain_mask, phage_mask)

            # Use batch-specific weights in the loss calculation if enabled
            if use_phage_weights:
                criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
            else:
                criterion = nn.BCEWithLogitsLoss()

            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(val_loader)
    binary_preds = (np.array(all_preds) > 0.5).astype(int)
    binary_labels = (np.array(all_labels) > 0.5).astype(int)
    val_mcc = matthews_corrcoef(binary_labels, binary_preds)

    return val_loss, val_mcc, binary_preds, binary_labels


def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                trial: Optional[Any] = None,
                num_epochs: int = 100,
                learning_rate: float = 5e-5,
                patience: int = 7,
                device: Optional[torch.device] = None,
                use_phage_weights: bool = True,
                scheduler_type: str = "one_cycle",
                warmup_ratio: float = 0.1,
                weight_decay: float = 0.01,
                metrics_dir: Optional[str] = None) -> Tuple[Dict[str, List[float]], float]:
    """
    Train model with configurable parameters and optional Optuna pruning.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        trial: Optional Optuna trial for pruning
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        device: Device to train on
        use_phage_weights: Whether to use phage-specific weights
        scheduler_type: Type of learning rate scheduler
        warmup_ratio: Warmup ratio for scheduler
        weight_decay: Weight decay for optimizer
        metrics_dir: Directory to save metrics
        
    Returns:
        Tuple of (history, best_val_mcc)
    """
    if device is None:
        device = get_device()
        
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Setup scheduler based on type
    num_train_steps = len(train_loader) * num_epochs
    scheduler = get_scheduler(
        scheduler_type,
        optimizer,
        num_train_steps,
        warmup_ratio
    )

    early_stopping = EarlyStopping(patience=patience, mode='max')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mcc': [],
        'val_mcc': [],
        'lr': []
    }

    metrics_records = []

    try:
        for epoch in range(num_epochs):
            # Train and validate
            train_loss, train_mcc = train_epoch(
                model, train_loader, optimizer, device,
                scheduler, use_phage_weights
            )
            val_loss, val_mcc, _, _ = validate(
                model, val_loader, device, use_phage_weights
            )

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mcc'].append(train_mcc)
            history['val_mcc'].append(val_mcc)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            # Add to metrics records
            metrics_records.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mcc': train_mcc,
                'val_mcc': val_mcc,
                'lr': optimizer.param_groups[0]['lr']
            })

            # Log epoch summary
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, MCC: {train_mcc:.4f} | "
                       f"Val Loss: {val_loss:.4f}, MCC: {val_mcc:.4f} | "
                       f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Update ReduceLROnPlateau scheduler if used
            if scheduler_type == "reduce_on_plateau" and scheduler is not None:
                scheduler.step(val_mcc)

            # Report to Optuna for pruning
            if trial is not None:
                trial.report(val_mcc, epoch)

                # Handle pruning based on val_mcc
                if trial.should_prune():
                    logger.info("Trial pruned by Optuna.")
                    raise optuna.exceptions.TrialPruned()

            # Check early stopping
            best_val_mcc = early_stopping(val_mcc, model)
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}. "
                           f"Best val MCC: {best_val_mcc:.4f}")
                break

    except Exception as e:
        if hasattr(e, '__class__') and e.__class__.__name__ == 'TrialPruned':
            raise e
        else:
            logger.error(f"Training error: {e}")
            raise

    # Save metrics to CSV
    if metrics_dir:
        trial_id = trial.number if trial is not None else "final"
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_df = pd.DataFrame(metrics_records)
        metrics_df.to_csv(os.path.join(metrics_dir, f"trial_{trial_id}_metrics.csv"), index=False)

    # Restore best model state if available
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    return history, best_val_mcc if early_stopping.best_metric is not None else val_mcc


def train_model_with_params(interactions_path: str,
                           strain_embeddings_path: str,
                           phage_embeddings_path: str,
                           output_dir: Optional[str] = None,
                           # Model parameters
                           embedding_dim: Optional[int] = None,
                           hidden_dim: int = 512,
                           num_heads: int = 8,
                           strain_inds: int = 256,
                           phage_inds: int = 256,
                           num_isab_layers: int = 2,
                           num_seeds: int = 1,
                           dropout: float = 0.1,
                           ln: bool = True,
                           temperature: float = 0.1,
                           use_cross_attention: bool = True,
                           classifier_hidden_layers: int = 1,
                           classifier_hidden_dim: int = 512,
                           activation_function: str = "gelu",
                           chunk_size: int = 128,
                           normalization_type: str = "none",
                           # Training parameters
                           num_epochs: int = 100,
                           learning_rate: float = 1e-4,
                           batch_size: int = 64,
                           patience: int = 10,
                           use_phage_weights: bool = True,
                           weight_decay: float = 0.01,
                           scheduler_type: str = "one_cycle",
                           warmup_ratio: float = 0.1,
                           random_state: int = 42,
                           return_attention: bool = True,
                           log_level: str = "INFO") -> Dict[str, Any]:
    """
    Train a model with fixed hyperparameters.
    
    This function trains a FlexibleStrainPhageTransformer model with the specified
    hyperparameters and saves all outputs including the model, metrics, and plots.
    
    Args:
        interactions_path: Path to CSV with interaction data
        strain_embeddings_path: Path to directory with strain embeddings
        phage_embeddings_path: Path to directory with phage embeddings
        output_dir: Output directory (if None, creates timestamped directory)
        
        Model architecture parameters:
        embedding_dim: Embedding dimension (if None, inferred from data)
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        strain_inds: Number of inducing points for strains
        phage_inds: Number of inducing points for phages
        num_isab_layers: Number of ISAB layers
        num_seeds: Number of seeds for PMA
        dropout: Dropout rate
        ln: Whether to use layer normalization
        temperature: Temperature for attention
        use_cross_attention: Whether to use cross-attention
        classifier_hidden_layers: Number of hidden layers in classifier
        classifier_hidden_dim: Hidden dimension for classifier
        activation_function: Activation function ("relu", "gelu", "silu")
        chunk_size: Chunk size for attention computation
        
        Training parameters:
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        batch_size: Batch size
        patience: Early stopping patience
        use_phage_weights: Whether to use phage-specific weights
        weight_decay: Weight decay
        scheduler_type: LR scheduler type
        warmup_ratio: Warmup ratio for scheduler
        random_state: Random seed
        return_attention: Whether to save attention weights
        log_level: Logging level
        
    Returns:
        Dictionary containing:
        - 'model_path': Path to saved model
        - 'metrics': Test set metrics
        - 'history': Training history
        - 'output_dir': Output directory path
    """
    # Setup logging
    setup_logging(level=log_level)
    
    # Create output directory
    if output_dir is None:
        output_dir = create_output_directory(prefix="pst_train")
    else:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    
    # Setup logging to file
    log_file = os.path.join(output_dir, "training.log")
    setup_logging(level=log_level, log_file=log_file)
    
    logger.info("Starting model training with fixed parameters")
    logger.info(f"Output directory: {output_dir}")
    
    # Get device
    device = get_device()
    
    # Load data
    logger.info("Loading embeddings...")
    strain_embeddings = load_embeddings_flexible(strain_embeddings_path)
    phage_embeddings = load_embeddings_flexible(phage_embeddings_path)
    
    logger.info("Loading interaction data...")
    interactions_df = pd.read_csv(interactions_path)
    
    # Filter to ensure we have embeddings for all strains/phages
    strain_keys = set(strain_embeddings.keys())
    phage_keys = set(phage_embeddings.keys())
    
    filtered_df = interactions_df[
        interactions_df['strain'].isin(strain_keys) &
        interactions_df['phage'].isin(phage_keys)
    ]
    
    logger.info(f"Original interactions: {len(interactions_df)}")
    logger.info(f"Filtered interactions: {len(filtered_df)}")
    
    # Split data
    train_df, test_df = filter_interactions_by_strain(filtered_df, random_state)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_df, test_df, strain_embeddings, phage_embeddings,
        batch_size=batch_size, use_phage_weights=use_phage_weights
    )
    
    # Get embedding dimension if not specified
    if embedding_dim is None:
        first_strain_id = next(iter(strain_embeddings))
        embedding_dim = strain_embeddings[first_strain_id][0].shape[1]
        logger.info(f"Inferred embedding dimension: {embedding_dim}")
    
    # Initialize model
    model = FlexibleStrainPhageTransformer(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        strain_inds=strain_inds,
        phage_inds=phage_inds,
        num_isab_layers=num_isab_layers,
        num_seeds=num_seeds,
        dropout=dropout,
        ln=ln,
        temperature=temperature,
        use_cross_attention=use_cross_attention,
        classifier_hidden_layers=classifier_hidden_layers,
        classifier_hidden_dim=classifier_hidden_dim,
        activation_function=activation_function,
        chunk_size=chunk_size,
        normalization_type=normalization_type 
    ).to(device)
    
    # Initialize attention weights
    model = init_attention_weights(model)
    
    # Save configuration
    config = {
        'data': {
            'interactions_path': interactions_path,
            'strain_embeddings_path': strain_embeddings_path,
            'phage_embeddings_path': phage_embeddings_path,
            'train_size': len(train_df),
            'test_size': len(test_df),
        },
        'model': {
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'strain_inds': strain_inds,
            'phage_inds': phage_inds,
            'num_isab_layers': num_isab_layers,
            'num_seeds': num_seeds,
            'dropout': dropout,
            'ln': ln,
            'temperature': temperature,
            'use_cross_attention': use_cross_attention,
            'classifier_hidden_layers': classifier_hidden_layers,
            'classifier_hidden_dim': classifier_hidden_dim,
            'activation_function': activation_function,
            'chunk_size': chunk_size,
            'normalization_type': normalization_type,
        },
        'training': {
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'patience': patience,
            'use_phage_weights': use_phage_weights,
            'weight_decay': weight_decay,
            'scheduler_type': scheduler_type,
            'warmup_ratio': warmup_ratio,
            'random_state': random_state,
        }
    }
    save_config(config, output_dir)
    
    # Train model
    logger.info("Starting training...")
    history, best_val_mcc = train_model(
        model,
        train_loader,
        test_loader,
        trial=None,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
        use_phage_weights=use_phage_weights,
        scheduler_type=scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        metrics_dir=os.path.join(output_dir, "metrics")
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_full(
        model, test_loader, device, use_phage_weights,
        return_attention=return_attention,
        output_dir=output_dir
    )
    
    # Save model
    model_path = os.path.join(output_dir, "models", "model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_metrics': {k: v for k, v in test_metrics.items() 
                        if k not in ['all_preds', 'all_labels', 'binary_preds', 'conf_matrix', 'attention_weights']},
        'embedding_dim': embedding_dim
    }, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Plot training history
    plot_training_history(history, "final", output_dir)
    
    # Print final results
    logger.info("\n" + "="*80)
    logger.info("FINAL MODEL PERFORMANCE")
    logger.info("="*80)
    logger.info(f"Test MCC:       {test_metrics['mcc']:.4f}")
    logger.info(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Score:  {test_metrics['f1']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"Test ROC AUC:   {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test PR AUC:    {test_metrics['pr_auc']:.4f}")
    logger.info("="*80)
    
    return {
        'model_path': model_path,
        'metrics': test_metrics,
        'history': history,
        'output_dir': output_dir
    }
