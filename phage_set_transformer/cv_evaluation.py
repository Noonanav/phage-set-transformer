"""
CV evaluation with fixed hyperparameters for the phage-set-transformer package.
"""

import json
import logging
import os
from pathlib import Path
import yaml
from typing import Dict, Any, Optional

import pandas as pd
import torch

# package-internal imports  
from .data import load_embeddings_flexible, create_data_loaders
from .models import FlexibleStrainPhageTransformer, init_attention_weights
from .training import train_model
from .evaluation import evaluate_full
from .utils import setup_logging, create_output_directory, NumpyEncoder, get_device
from .optimization import _split_by_strain, _model_kw

_log = logging.getLogger(__name__)


def load_fixed_params(config_path: str) -> Dict[str, Any]:
    """Load fixed hyperparameters from YAML file."""
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    _log.info(f"Loaded fixed parameters from {config_path}")
    return params


def run_single_fold_seed(
    interactions_path: str,
    strain_embeddings_path: str, 
    phage_embeddings_path: str,
    params_config_path: str,
    job_id: int,
    n_folds: int = 5,
    seeds_per_fold: int = 3,
    base_seed: int = 42,
    output_dir: str = "cv_results",
    num_epochs: int = 150,
    patience: int = 15,
    return_attention: bool = True,
) -> None:
    """
    Run single fold×seed combination for SLURM array job.
    
    Args:
        job_id: SLURM array task ID
        Other args: Same as optimization workflow
    """
    # Calculate fold and seed from job_id
    fold_id = job_id // seeds_per_fold
    seed_offset = job_id % seeds_per_fold
    actual_seed = base_seed + fold_id * 1000 + seed_offset
    
    _log.info(f"Job {job_id}: fold {fold_id}, seed offset {seed_offset}, actual seed {actual_seed}")
    
    # Load data
    fixed_params = load_fixed_params(params_config_path)
    interactions = pd.read_csv(interactions_path)
    strain_embeddings = load_embeddings_flexible(strain_embeddings_path)
    phage_embeddings = load_embeddings_flexible(phage_embeddings_path)
    
    # Filter interactions
    valid_interactions = interactions[
        interactions["strain"].isin(strain_embeddings.keys()) &
        interactions["phage"].isin(phage_embeddings.keys())
    ]
    
    # Create this specific fold split (reuse existing logic)
    train_df, test_df = _split_by_strain(valid_interactions, actual_seed)
    
    # Create output directory for this run
    run_dir = Path(output_dir) / f"fold_{fold_id}_seed_{actual_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_df, test_df, strain_embeddings, phage_embeddings,
        batch_size=fixed_params["batch_size"],
        use_phage_weights=fixed_params["use_phage_weights"],
        random_state=actual_seed
    )
    
    # Create model (reuse existing logic)
    device = get_device()
    emb_dim = next(iter(strain_embeddings.values()))[0].shape[1]
    model_params = {k: v for k, v in fixed_params.items() if k in _model_kw()}
    
    model = FlexibleStrainPhageTransformer(
        embedding_dim=emb_dim,
        hidden_dim=512,
        classifier_hidden_dim=512, 
        chunk_size=128,
        **model_params
    )
    model = init_attention_weights(model).to(device)
    
    # Train model (reuse existing function)
    _, best_val_mcc = train_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs,
        learning_rate=fixed_params["learning_rate"],
        patience=patience,
        device=device,
        use_phage_weights=fixed_params["use_phage_weights"],
        scheduler_type=fixed_params["scheduler_type"],
        warmup_ratio=fixed_params["warmup_ratio"],
        weight_decay=fixed_params["weight_decay"],
    )
    
    # Evaluate model (reuse existing function)
    metrics = evaluate_full(
        model, val_loader, device,
        fixed_params["use_phage_weights"],
        return_attention=return_attention,
        output_dir=str(run_dir)
    )
    
    # Save model (reuse existing pattern)
    model_config = {'embedding_dim': emb_dim, 'hidden_dim': 512, 'classifier_hidden_dim': 512, 'chunk_size': 128, **model_params}
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'model': model_config},
        'fold_id': fold_id,
        'seed': actual_seed,
        'metrics': {k: v for k, v in metrics.items() if k not in ['all_preds', 'all_labels', 'binary_preds', 'conf_matrix', 'attention_weights']},
        'fixed_params': fixed_params,
    }, run_dir / "model.pt")
    
    # Save simple summary
    summary = {
        'fold_id': fold_id,
        'seed': actual_seed, 
        'final_mcc': metrics['mcc'],
        'train_size': len(train_df),
        'val_size': len(test_df),
    }
    
    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    _log.info(f"Completed fold {fold_id}, seed {actual_seed}. MCC: {metrics['mcc']:.4f}")


def run_cv_evaluation(
    interactions_path: str,
    strain_embeddings_path: str,
    phage_embeddings_path: str, 
    params_config_path: str,
    n_folds: int = 5,
    seeds_per_fold: int = 3,
    base_seed: int = 42,
    output_dir: Optional[str] = None,
    **kwargs
) -> None:
    """
    Run complete CV evaluation locally.
    
    Just runs all fold×seed combinations sequentially.
    """
    if output_dir is None:
        output_dir = create_output_directory(prefix="cv_evaluation")
    
    # Ensure output_dir is a string for compatibility
    output_dir = str(output_dir)
    
    total_jobs = n_folds * seeds_per_fold
    _log.info(f"Running {total_jobs} CV jobs ({n_folds} folds × {seeds_per_fold} seeds)")
    
    for job_id in range(total_jobs):
        _log.info(f"Running job {job_id+1}/{total_jobs}")
        run_single_fold_seed(
            interactions_path, strain_embeddings_path, phage_embeddings_path,
            params_config_path, job_id, n_folds, seeds_per_fold, base_seed, 
            output_dir, **kwargs
        )
    
    _log.info(f"CV evaluation complete. Results in: {output_dir}")
