"""
optimization.py  ·  Cross-validated hyper-parameter search
=========================================================
This module replaces the single-split optimisation with *k*-fold CV and an
optional final multi-seed retraining step, mirroring the notebook
`set_transformer_optuna_optimization_cv`:contentReference[oaicite:0]{index=0}.

Public API
----------
run_cv_optimization(...)      # five-fold CV (default) Optuna study
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import logging
import os
from pathlib import Path
import yaml
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold
from optuna.samplers import TPESampler
from optuna.pruners import PercentilePruner
from optuna.storages import RDBStorage
import torch

# package-internal imports
from .data import create_data_loaders
from .models import FlexibleStrainPhageTransformer, init_attention_weights
from .training import train_model
from .evaluation import evaluate_full
from .utils import setup_logging, create_output_directory, NumpyEncoder

_log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# Hyper-parameter search space
# -----------------------------------------------------------------------------#
def _suggest_params(trial: optuna.Trial, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return one sampled hyper-parameter set."""
    if config is None:
        # Original hardcoded defaults - exactly as before
        return dict(
            # architecture
            use_cross_attention=trial.suggest_categorical("use_cross_attention", [True, False]),
            num_heads=trial.suggest_categorical("num_heads", [2, 4, 8]),
            temperature=trial.suggest_float("temperature", 0.01, 1.0, log=True),
            strain_inds=trial.suggest_categorical("strain_inds", [64, 128, 192, 256]),
            phage_inds=trial.suggest_categorical("phage_inds", [32, 64, 96, 128]),
            num_isab_layers=trial.suggest_int("num_isab_layers", 1, 3),
            ln=trial.suggest_categorical("ln", [True, False]),
            num_seeds=trial.suggest_categorical("num_seeds", [1, 2, 4]),
            normalization_type=trial.suggest_categorical("normalization_type", ["none", "layer_norm", "l2_norm"]),
            use_residual_classifier=trial.suggest_categorical("use_residual_classifier", [False, True]),
            # classifier
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            classifier_hidden_layers=trial.suggest_int("classifier_hidden_layers", 2, 6),
            activation_function=trial.suggest_categorical("activation_function", ["relu", "gelu", "silu"]),
            # training
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            use_phage_weights=trial.suggest_categorical("use_phage_weights", [True, False]),
            weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
            # scheduler
            scheduler_type=trial.suggest_categorical(
                "scheduler_type",
                ["one_cycle", "cosine_annealing", "reduce_on_plateau", "linear_warmup_decay", "none"],
            ),
            warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
        )
    
    # Use config if provided
    params = {}
    for param_name, param_config in config.items():
        if param_config["type"] == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])
        elif param_config["type"] == "float":
            log_scale = param_config.get("log", False)
            params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"], log=log_scale)
        elif param_config["type"] == "int":
            params[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"])
    
    return params

# -----------------------------------------------------------------------------#
# CV objective
# -----------------------------------------------------------------------------#
def _cv_objective(
    trial: optuna.Trial,
    *,
    interactions: pd.DataFrame,
    strain_embeddings: Dict[str, np.ndarray],
    phage_embeddings: Dict[str, np.ndarray],
    n_folds: int,
    random_state: int,
    base_trial_dir: Path,
    search_config: Optional[Dict[str, Any]] = None,
    cv_epochs: int = 50,
    cv_patience: int = 7,
    val_batch_size: Optional[int] = None,
    stability_min_epoch: int = 8,
    stability_loss_margin: float = 0.15,
    stability_loss_lookback: int = 8,
    accumulation_steps: int = 4,
) -> float:
    """Return **stability-aware median AUPR** across *n_folds* folds (prunable)."""
    params = _suggest_params(trial, search_config) 
    fold_aupr: List[float] = []

    from .utils import get_device
    device = get_device()

    # stratify by strain id so no strain leaks across folds
    unique_strains = interactions["strain"].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(unique_strains)):
        train_strains = unique_strains[train_idx]
        val_strains = unique_strains[val_idx]

        train_df = interactions[interactions["strain"].isin(train_strains)]
        val_df = interactions[interactions["strain"].isin(val_strains)]

        train_loader, val_loader = create_data_loaders(
            train_df,
            val_df,
            strain_embeddings,
            phage_embeddings,
            batch_size=params["batch_size"],
            val_batch_size=val_batch_size, 
            use_phage_weights=params["use_phage_weights"],
            random_state=random_state + fold_idx 
        )

        # infer embedding dim from first strain vector
        emb_dim = next(iter(strain_embeddings.values()))[0].shape[1]

        model = FlexibleStrainPhageTransformer(
            embedding_dim=emb_dim,
            num_heads=params["num_heads"],
            strain_inds=params["strain_inds"],
            phage_inds=params["phage_inds"],
            num_isab_layers=params["num_isab_layers"],
            num_seeds=params["num_seeds"],
            dropout=params["dropout"],
            ln=params["ln"],
            temperature=params["temperature"],
            use_cross_attention=params["use_cross_attention"],
            classifier_hidden_layers=params["classifier_hidden_layers"],
            activation_function=params["activation_function"],
            normalization_type=params["normalization_type"],
            use_residual_classifier=params["use_residual_classifier"],
        )
        model = init_attention_weights(model)
        model = model.to(device)

        history, _val_aupr = train_model(
            model,
            train_loader,
            val_loader,
            trial=None,  # Don't pass trial to avoid double reporting
            num_epochs=cv_epochs,
            learning_rate=params["learning_rate"],
            patience=cv_patience, 
            use_phage_weights=params["use_phage_weights"],
            scheduler_type=params["scheduler_type"],
            warmup_ratio=params["warmup_ratio"],
            weight_decay=params["weight_decay"],
            device=device,
            accumulation_steps=accumulation_steps, 
        )

        # Apply stability filtering to fold result
        stable_aupr = _get_stable_best_aupr(
            history['val_loss'],
            history['val_aupr'],
            min_epoch=stability_min_epoch,
            loss_margin=stability_loss_margin,
            loss_lookback=stability_loss_lookback
        )
        fold_aupr.append(stable_aupr) 

        # Report stability-filtered AUPR for pruning decisions
        trial.report(stable_aupr, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    median_aupr = float(np.median(fold_aupr))
    # lightweight report artefact
    with (base_trial_dir / f"trial_{trial.number}_fold_aupr.json").open("w") as f:
        _json.dump({"fold_aupr": fold_aupr, "median_aupr": median_aupr}, f, indent=2)

    return median_aupr


# -----------------------------------------------------------------------------#
# Public entry-point
# -----------------------------------------------------------------------------#
def run_cv_optimization(
    *,
    interactions_path: str,
    strain_embeddings_path: str,
    phage_embeddings_path: str,
    n_trials: int = 200,
    n_folds: int = 5,
    final_seeds: int = 5,
    cv_epochs: int = 100,
    cv_patience: int = 7, 
    random_state: int = 42,
    study_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    log_level: str = "INFO",
    search_config_path: Optional[str] = None,
    val_batch_size: Optional[int] = None,
    stability_min_epoch: int = 8,
    stability_loss_margin: float = 0.08,
    stability_loss_lookback: int = 5,
    accumulation_steps: int = 4, 
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Run an Optuna hyper-parameter search **with k-fold CV per trial** and then
    retrain the best configuration with several different random seeds.  
    Returns the Optuna study and a dict containing final model artefacts.

    Set *final_seeds=0* if you do **not** want the multi-seed retrain step.
    """
    setup_logging(level=log_level)

    # Load config if provided
    search_config = None
    if search_config_path:
        with open(search_config_path, 'r') as f:
            search_config = yaml.safe_load(f)
        _log.info(f"Loaded search configuration from {search_config_path}")

    # ---------------------------------------------------------------- data
    interactions = pd.read_csv(interactions_path)
    strain_embeddings = _load_embeddings(strain_embeddings_path)
    phage_embeddings = _load_embeddings(phage_embeddings_path)

    # keep only interactions for which embeddings exist
    valid = interactions[
        interactions["strain"].isin(strain_embeddings.keys())
        & interactions["phage"].isin(phage_embeddings.keys())
    ]
    if valid.empty:
        raise ValueError("No interactions overlap with provided embeddings")

    _log.info("Interactions after filtering: %d", len(valid))

    # ---------------------------------------------------------------- dirs / Optuna storage
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = (
        Path(output_dir)
        if output_dir
        else create_output_directory(prefix="optuna_cv_runs", timestamp=ts)
    )
    trial_dir = base_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    storage = RDBStorage(url=f"sqlite:///{base_dir/'study.db'}")
    study_name = study_name or f"cv_search_{ts}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        _log.info("Resuming study '%s' with %d completed trials", study_name, len(study.trials))
    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(seed=random_state),
            pruner=PercentilePruner(percentile=25.0, n_startup_trials=3, n_warmup_steps=1),
            storage=storage,
        )
        _log.info("Created new study '%s'", study_name)

    # ---------------------------------------------------------------- optimisation
    if n_trials == 1:
        # Single trial mode (likely SLURM) - always run the requested trial
        trials_to_run = 1
    else:
        # Batch mode - use resume logic  
        trials_to_run = max(0, n_trials - len(study.trials))

    study.optimize(
        lambda t: _cv_objective(
            t,
            interactions=valid,
            strain_embeddings=strain_embeddings,
            phage_embeddings=phage_embeddings,
            n_folds=n_folds,
            random_state=random_state,
            base_trial_dir=trial_dir,
            search_config=search_config,
            cv_epochs=cv_epochs,
            cv_patience=cv_patience,
            val_batch_size=val_batch_size,
            stability_min_epoch=stability_min_epoch,
            stability_loss_margin=stability_loss_margin,
            stability_loss_lookback=stability_loss_lookback,
            accumulation_steps=accumulation_steps,
        ),
        n_trials=trials_to_run,
        show_progress_bar=True,
    )

    if len(study.trials) > 0 and any(trial.value is not None for trial in study.trials):
        _log.info("Best median AUPR = %.4f", study.best_value)
    else:
        _log.info("No successful trials yet (all trials pruned or failed)")

    # ---------------------------------------------------------------- optional multi-seed retrain
    final_summary = {}
    if final_seeds > 0 and len(study.trials) > 0 and any(trial.value is not None for trial in study.trials):
        final_summary = _retrain_best_params(
            study,
            interactions=valid,
            strain_embeddings=strain_embeddings,
            phage_embeddings=phage_embeddings,
            n_runs=final_seeds,
            base_dir=base_dir,
            random_state=random_state,
            val_batch_size=val_batch_size,
            accumulation_steps=accumulation_steps, 
        )
        _log.info("Final retraining completed")
    else:
        _log.info("Skipping final retraining - no successful trials to retrain")

    # save study artefacts
    study.trials_dataframe().to_csv(base_dir / "all_trials.csv", index=False)
    if len(study.trials) > 0 and any(trial.value is not None for trial in study.trials):
        with (base_dir / "best_params.json").open("w") as f:
            _json.dump(study.best_params, f, indent=2)
    else:
        _log.warning("No successful trials to save best parameters")

    return study, final_summary


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _load_embeddings(folder: str) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Load *.npy* embedding files into a dict keyed by basename."""
    from .data import load_embeddings_flexible
    return load_embeddings_flexible(folder)

def _retrain_best_params(
    study: optuna.Study,
    *,
    interactions: pd.DataFrame,
    strain_embeddings: Dict[str, Tuple[np.ndarray, List[str]]],
    phage_embeddings: Dict[str, Tuple[np.ndarray, List[str]]],
    n_runs: int,
    base_dir: Path,
    random_state: int,
    val_batch_size: Optional[int] = None,
    accumulation_steps: int = 4, 
) -> Dict[str, Any]:
    """
    Retrain best hyper-parameter set `n_runs` times with different seeds,
    returning aggregate metrics (median, mean ± std) exactly like the notebook
    implementation :contentReference[oaicite:1]{index=1}.
    """

    best_params = study.best_params
    all_metrics = []
    models = []

    # CREATE THE NECESSARY DIRECTORIES
    final_models_dir = base_dir / "final_models"
    final_models_dir.mkdir(parents=True, exist_ok=True)

    from .utils import get_device
    device = get_device()

    for i in range(n_runs):
        seed = random_state + i
        train_df, test_df = _split_by_strain(interactions, seed)

        train_loader, test_loader = create_data_loaders(
            train_df,
            test_df,
            strain_embeddings,
            phage_embeddings,
            batch_size=best_params["batch_size"],
            val_batch_size=val_batch_size, 
            use_phage_weights=best_params["use_phage_weights"],
            random_state=seed 
        )

        emb_dim = next(iter(strain_embeddings.values()))[0].shape[1]
        model = FlexibleStrainPhageTransformer(
            embedding_dim=emb_dim,
            **{k: v for k, v in best_params.items() if k in _model_kw()}
        )
        model = init_attention_weights(model)

        model = model.to(device)

        train_model(
            model,
            train_loader,
            test_loader,
            trial=None,
            num_epochs=150,
            learning_rate=best_params["learning_rate"],
            patience=15,
            use_phage_weights=best_params["use_phage_weights"],
            scheduler_type=best_params["scheduler_type"],
            warmup_ratio=best_params["warmup_ratio"],
            weight_decay=best_params["weight_decay"],
            device=device,               # default cuda/auto
            metrics_dir=str(base_dir / "training_metrics"),
            accumulation_steps=accumulation_steps, 
        )

        # CREATE THE SEED-SPECIFIC EVALUATION DIRECTORY
        seed_eval_dir = final_models_dir / f"seed_{seed}_evaluation"
        seed_eval_dir.mkdir(parents=True, exist_ok=True)

        metrics = evaluate_full(
            model, 
            test_loader, 
            device, 
            best_params["use_phage_weights"],
            return_attention=True,
            output_dir=str(seed_eval_dir))  # Use the created directory
        all_metrics.append(metrics)
        models.append(model)

        full_model_config = {
            'embedding_dim': emb_dim,
            'hidden_dim': 512,  # Add the default used
            'num_heads': best_params['num_heads'],
            'strain_inds': best_params['strain_inds'], 
            'phage_inds': best_params['phage_inds'],
            'num_isab_layers': best_params['num_isab_layers'],
            'num_seeds': best_params['num_seeds'],
            'dropout': best_params['dropout'],
            'ln': best_params['ln'],
            'temperature': best_params['temperature'],
            'use_cross_attention': best_params['use_cross_attention'],
            'classifier_hidden_layers': best_params['classifier_hidden_layers'],
            'classifier_hidden_dim': 512,  # Add the default used
            'activation_function': best_params['activation_function'],
            'chunk_size': 128,   # Add the default used
            'normalization_type': best_params['normalization_type'],
            'use_residual_classifier': best_params['use_residual_classifier'],
        }

        # persist each seed model
        torch_save = dict(
            model_state_dict=model.state_dict(),
            config={'model': full_model_config}, 
            seed=seed,
            embedding_dim=emb_dim,
            metrics=metrics,
            best_params=best_params,
        )
        torch.save(torch_save, final_models_dir / f"model_seed_{seed}.pt")

    auprs = [m["pr_auc"] for m in all_metrics]
    summary = dict(
        median_aupr=float(np.median(auprs)),
        mean_aupr=float(np.mean(auprs)),
        std_aupr=float(np.std(auprs)),
        per_seed_metrics=all_metrics,
    )
    with (base_dir / "multi_seed_summary.json").open("w") as f:
        _json.dump(summary, f, indent=2, cls=NumpyEncoder)
    return summary


def _split_by_strain(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    80/20 split stratified by strain id with class balance optimization.
    
    Tries multiple random strain splits to find one that maximizes the minimum
    number of positive examples in both train and test sets, while maintaining
    strain stratification (no strain appears in both splits).
    
    Args:
        df: DataFrame with columns 'strain', 'phage', 'interaction'
        seed: Base random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df) with optimized class balance
    """
    strains = df["strain"].unique()
    
    # Try multiple random seeds to find balanced split
    best_train_df, best_test_df = None, None
    best_min_pos = 0
    
    for attempt in range(50):  # Fewer attempts since this runs many times in optimization
        rng = np.random.RandomState(seed + attempt)
        shuffled_strains = strains.copy()
        rng.shuffle(shuffled_strains)

        cut = int(0.8 * len(shuffled_strains))
        train_strains = shuffled_strains[:cut]
        test_strains = shuffled_strains[cut:]
        
        train_df = df[df["strain"].isin(train_strains)]
        test_df = df[df["strain"].isin(test_strains)]
        
        train_pos = train_df['interaction'].sum()
        test_pos = test_df['interaction'].sum()
        
        # Keep track of split with best minimum positive count
        min_pos = min(train_pos, test_pos)
        if min_pos > best_min_pos:
            best_min_pos = min_pos
            best_train_df = train_df
            best_test_df = test_df
            
            # If we have decent balance, stop early
            if min_pos >= 5:  # Lower threshold for optimization runs
                break
    
    # Fallback to original approach if no good split found
    if best_train_df is None:
        rng = np.random.RandomState(seed)
        shuffled_strains = strains.copy()
        rng.shuffle(shuffled_strains)
        
        cut = int(0.8 * len(shuffled_strains))
        train_strains = shuffled_strains[:cut]
        best_train_df = df[df["strain"].isin(train_strains)]
        best_test_df = df[~df["strain"].isin(train_strains)]
    
    return best_train_df, best_test_df

def _get_stable_best_aupr(
    loss_history: List[float], 
    aupr_history: List[float],
    min_epoch: int = 8,
    loss_margin: float = 0.08,
    loss_lookback: int = 5,
    early_epoch_window: int = 3
) -> float:
    """
    Select best AUPR from epochs with stable/decreasing validation loss.
    
    This function filters training epochs to select only those that show good
    generalization (low loss) without overfitting. It uses multiple criteria:
    1. Loss must be near the global minimum (within margin)
    2. Loss must not exceed early training baseline (no severe overfitting)
    3. Loss must not be on a sharp upward trend (if enough history exists)
    
    The function returns the maximum AUPR among epochs passing all filters.
    If no epochs pass, returns 0.0 (will likely trigger Optuna pruning).
    
    Args:
        loss_history: List of validation losses, one per epoch
        aupr_history: List of validation AUPRs, one per epoch (same length as loss_history)
        min_epoch: Minimum epoch to consider (skip noisy early training).
            Must have at least (min_epoch + early_epoch_window) epochs total.
        loss_margin: Allowed fractional increase above global minimum loss.
            E.g., 0.08 means accept losses up to 8% above the best achieved.
        loss_lookback: Number of recent epochs to check for upward trends.
            If current loss is >10% above recent average, epoch is rejected.
        early_epoch_window: Number of epochs to use for early baseline calculation.
            Median of these epochs establishes the "no worse than early training" threshold.
    
    Returns:
        Maximum AUPR from stable epochs, or 0.0 if no epochs meet criteria.
    """
    # Ensure we have enough epochs for meaningful analysis
    if len(loss_history) < min_epoch + early_epoch_window:
        return 0.0
    
    # STEP 1: Establish early training baseline
    # Use median of first few post-min_epoch epochs to detect overfitting
    early_start = min_epoch
    early_end = min_epoch + early_epoch_window
    early_baseline = float(np.median(loss_history[early_start:early_end]))
    
    # STEP 2: Find global minimum loss and set threshold
    global_min_loss = min(loss_history[min_epoch:])
    loss_threshold = global_min_loss * (1 + loss_margin)
    
    # STEP 3: Evaluate each epoch against all criteria
    stable_auprs = []
    
    for epoch in range(min_epoch, len(loss_history)):
        current_loss = loss_history[epoch]
        
        # Criterion 1: Loss must be near optimal (within margin of global minimum)
        if current_loss > loss_threshold:
            continue
            
        # Criterion 2: Loss must not be worse than early training (overfitting check)
        if current_loss > early_baseline:
            continue
        
        # Criterion 3: Loss must not be on sharp upward trend (if enough history)
        if epoch >= min_epoch + loss_lookback:
            recent_losses = loss_history[epoch - loss_lookback:epoch]
            recent_mean = float(np.mean(recent_losses))
            
            # Reject if current loss is >10% higher than recent average
            if current_loss > recent_mean * 1.10:
                continue
        
        # All criteria passed - this epoch has stable performance
        stable_auprs.append(aupr_history[epoch])
    
    # Return best AUPR from stable epochs, or 0.0 if none qualify
    return max(stable_auprs) if stable_auprs else 0.0


def _model_kw() -> set:
    """Return names that belong to model constructor (intersection helper)."""
    return {
        "num_heads",
        "strain_inds",
        "phage_inds",
        "num_isab_layers",
        "num_seeds",
        "dropout",
        "ln",
        "temperature",
        "use_cross_attention",
        "classifier_hidden_layers",
        "activation_function",
        "normalization_type",
        "use_residual_classifier",
    }
