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
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage

# package-internal imports
from .data import create_data_loaders
from .models import FlexibleStrainPhageTransformer, init_attention_weights
from .training import train_model
from .evaluation import evaluate_full
from .utils import setup_logging, create_output_directory

_log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# Hyper-parameter search space
# -----------------------------------------------------------------------------#
def _suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Return one sampled hyper-parameter set."""
    return dict(
        # architecture
        use_cross_attention=trial.suggest_categorical("use_cross_attention", [True, False]),
        num_heads=trial.suggest_categorical("num_heads", [4, 8, 16]),
        temperature=trial.suggest_float("temperature", 0.01, 1.0, log=True),
        strain_inds=trial.suggest_categorical("strain_inds", [64, 128, 192, 256]),
        phage_inds=trial.suggest_categorical("phage_inds", [32, 64, 96, 128]),
        num_isab_layers=trial.suggest_int("num_isab_layers", 1, 3),
        ln=trial.suggest_categorical("ln", [True, False]),
        num_seeds=trial.suggest_categorical("num_seeds", [1, 2, 4]),
        # classifier
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        classifier_hidden_layers=trial.suggest_int("classifier_hidden_layers", 1, 3),
        activation_function=trial.suggest_categorical("activation_function", ["relu", "gelu", "silu"]),
        # training
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
        use_phage_weights=trial.suggest_categorical("use_phage_weights", [True, False]),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
        # scheduler
        scheduler_type=trial.suggest_categorical(
            "scheduler_type",
            ["one_cycle", "cosine_annealing", "reduce_on_plateau", "linear_warmup_decay", "none"],
        ),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
    )


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
) -> float:
    """Return **median MCC** across *n_folds* folds (prunable)."""
    params = _suggest_params(trial)
    fold_mcc: List[float] = []

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
            use_phage_weights=params["use_phage_weights"],
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
        )
        model = init_attention_weights(model)

        _, _val_mcc = train_model(
            model,
            train_loader,
            val_loader,
            trial=None,                 # internal reporting disabled; we prune here
            num_epochs=75,
            learning_rate=params["learning_rate"],
            patience=7,
            use_phage_weights=params["use_phage_weights"],
            scheduler_type=params["scheduler_type"],
            warmup_ratio=params["warmup_ratio"],
            weight_decay=params["weight_decay"],
            device=None,               # default cuda/auto
        )

        fold_mcc.append(_val_mcc)
        trial.report(_val_mcc, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    median_mcc = float(np.median(fold_mcc))
    # lightweight report artefact – full pickle/report handled outside
    with (base_trial_dir / f"trial_{trial.number}_fold_mcc.json").open("w") as f:
        _json.dump({"fold_mcc": fold_mcc, "median_mcc": median_mcc}, f, indent=2)

    return median_mcc


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
    random_state: int = 42,
    study_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    log_level: str = "INFO",
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Run an Optuna hyper-parameter search **with k-fold CV per trial** and then
    retrain the best configuration with several different random seeds.  
    Returns the Optuna study and a dict containing final model artefacts.

    Set *final_seeds=0* if you do **not** want the multi-seed retrain step.
    """
    setup_logging(level=log_level)

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
            pruner=MedianPruner(n_startup_trials=3),
            storage=storage,
        )
        _log.info("Created new study '%s'", study_name)

    # ---------------------------------------------------------------- optimisation
    study.optimize(
        lambda t: _cv_objective(
            t,
            interactions=valid,
            strain_embeddings=strain_embeddings,
            phage_embeddings=phage_embeddings,
            n_folds=n_folds,
            random_state=random_state,
            base_trial_dir=trial_dir,
        ),
        n_trials=max(0, n_trials - len(study.trials)),
        show_progress_bar=True,
    )

    _log.info("Best median MCC = %.4f", study.best_value)

    # ---------------------------------------------------------------- optional multi-seed retrain
    final_summary = {}
    if final_seeds > 0:
        final_summary = _retrain_best_params(
            study,
            interactions=valid,
            strain_embeddings=strain_embeddings,
            phage_embeddings=phage_embeddings,
            n_runs=final_seeds,
            base_dir=base_dir,
            random_state=random_state,
        )

    # save study artefacts
    study.trials_dataframe().to_csv(base_dir / "all_trials.csv", index=False)
    with (base_dir / "best_params.json").open("w") as f:
        _json.dump(study.best_params, f, indent=2)

    return study, final_summary


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _load_embeddings(folder: str) -> Dict[str, np.ndarray]:
    """Load *.npy* embedding files into a dict keyed by basename."""
    data = {}
    for file in Path(folder).glob("*.npy"):
        data[file.stem] = np.load(file)
    if not data:
        raise FileNotFoundError(f"No .npy files found in {folder}")
    return data


def _retrain_best_params(
    study: optuna.Study,
    *,
    interactions: pd.DataFrame,
    strain_embeddings: Dict[str, np.ndarray],
    phage_embeddings: Dict[str, np.ndarray],
    n_runs: int,
    base_dir: Path,
    random_state: int,
) -> Dict[str, Any]:
    """
    Retrain best hyper-parameter set `n_runs` times with different seeds,
    returning aggregate metrics (median, mean ± std) exactly like the notebook
    implementation :contentReference[oaicite:1]{index=1}.
    """
    from .evaluation import metrics_to_dict  # assumes such util exists

    best_params = study.best_params
    all_metrics = []
    models = []

    for i in range(n_runs):
        seed = random_state + i
        train_df, test_df = _split_by_strain(interactions, seed)

        train_loader, test_loader = create_data_loaders(
            train_df,
            test_df,
            strain_embeddings,
            phage_embeddings,
            batch_size=best_params["batch_size"],
            use_phage_weights=best_params["use_phage_weights"],
        )

        emb_dim = next(iter(strain_embeddings.values()))[0].shape[1]
        model = FlexibleStrainPhageTransformer(
            embedding_dim=emb_dim,
            **{k: v for k, v in best_params.items() if k in _model_kw()}
        )
        model = init_attention_weights(model)

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
        )

        metrics = evaluate_full(model, test_loader, None, best_params["use_phage_weights"])
        all_metrics.append(metrics)
        models.append(model)

        # persist each seed model
        torch_save = dict(
            model_state_dict=model.state_dict(),
            seed=seed,
            embedding_dim=emb_dim,
            metrics=metrics,
            best_params=best_params,
        )
        torch.save(torch_save, base_dir / f"model_seed_{seed}.pt")

    mccs = [m["mcc"] for m in all_metrics]
    summary = dict(
        median_mcc=float(np.median(mccs)),
        mean_mcc=float(np.mean(mccs)),
        std_mcc=float(np.std(mccs)),
        per_seed_metrics=all_metrics,
    )
    with (base_dir / "multi_seed_summary.json").open("w") as f:
        _json.dump(summary, f, indent=2)
    return summary


def _split_by_strain(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split stratified by strain id."""
    strains = df["strain"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(strains)

    cut = int(0.8 * len(strains))
    train_strains = strains[:cut]
    train_df = df[df["strain"].isin(train_strains)]
    test_df = df[~df["strain"].isin(train_strains)]
    return train_df, test_df


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
    }
