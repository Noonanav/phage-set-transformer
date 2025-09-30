#!/usr/bin/env python
"""
Command-line interface for the phage-set-transformer package.

After installing the package (`pip install .`), run  
    pst --help
to view available sub-commands.
"""

import argparse
import logging
from pathlib import Path
import torch

# Package imports
from .training import train_model_with_params
from .optimization import run_cv_optimization
from .evaluation import predict
from .cv_evaluation import run_cv_evaluation, run_single_fold_seed, load_fixed_params


def _add_common_data_args(p):
    p.add_argument("--interactions", required=True,
                   help="CSV with columns strain, phage, interaction")
    p.add_argument("--strain-embeddings", required=True,
                   help="Directory of strain .npy files")
    p.add_argument("--phage-embeddings", required=True,
                   help="Directory of phage .npy files")


# ---------------------------------------------------------------------
# Sub-command: optimise
# ---------------------------------------------------------------------
def _cmd_optimize(args):
    study, _ = run_cv_optimization(
        interactions_path=args.interactions,
        strain_embeddings_path=args.strain_embeddings,
        phage_embeddings_path=args.phage_embeddings,
        n_trials=args.trials,
        n_folds=args.folds,
        final_seeds=args.final_seeds,
        random_state=args.seed,
        study_name=args.study_name,
        output_dir=getattr(args, 'output', None),
        search_config_path=getattr(args, 'search_config', None),
        cv_epochs=getattr(args, 'cv_epochs', 50),
        cv_patience=getattr(args, 'cv_patience', 7),
        val_batch_size=getattr(args, 'val_batch_size', None),
        stability_min_epoch=getattr(args, 'stability_min_epoch', 8),
        stability_loss_margin=getattr(args, 'stability_loss_margin', 0.15),
        stability_loss_lookback=getattr(args, 'stability_loss_lookback', 8),
        accumulation_steps=args.accumulation_steps, 
    )
    print(f"Best MCC = {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


# ---------------------------------------------------------------------
# Sub-command: train
# ---------------------------------------------------------------------
def _cmd_train(args):
    results = train_model_with_params(
        interactions_path=args.interactions,
        strain_embeddings_path=args.strain_embeddings,
        phage_embeddings_path=args.phage_embeddings,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        random_state=args.seed,
        normalization_type=args.normalization,
        use_residual_classifier=args.residual_classifier,
        val_batch_size=getattr(args, 'val_batch_size', None),
        accumulation_steps=args.accumulation_steps,
        # any extra kwargs you expose can be passed through here
    )
    print("Training finished; artefacts saved to", results["output_dir"])


# ---------------------------------------------------------------------
# Sub-command: predict
# ---------------------------------------------------------------------
def _cmd_predict(args):
    df = predict(
        model_path=args.model,
        strain_embeddings_path=args.strain_embeddings,
        phage_embeddings_path=args.phage_embeddings,
        interactions_path=args.pairs,
        batch_size=args.batch_size,
        return_attention=args.attention,
        device=torch.device(args.device),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Predictions saved to {out}")


# ---------------------------------------------------------------------
# Sub-command: cv-evaluate
# ---------------------------------------------------------------------
def _cmd_cv_evaluate(args):
    """Run cross-validation evaluation with fixed hyperparameters."""
    if hasattr(args, 'job_id') and args.job_id is not None:
        # SLURM array mode - run single fold×seed combination
        run_single_fold_seed(
            interactions_path=args.interactions,
            strain_embeddings_path=args.strain_embeddings,
            phage_embeddings_path=args.phage_embeddings,
            params_config_path=args.params_config,
            job_id=args.job_id,
            n_folds=args.folds,
            seeds_per_fold=args.seeds_per_fold,
            base_seed=args.seed,
            output_dir=args.output,
            num_epochs=args.epochs,
            patience=args.patience,
            return_attention=args.attention,
        )
        print(f"Completed job {args.job_id}")
    else:
        # Full evaluation mode
        run_cv_evaluation(
            interactions_path=args.interactions,
            strain_embeddings_path=args.strain_embeddings,
            phage_embeddings_path=args.phage_embeddings,
            params_config_path=args.params_config,
            n_folds=args.folds,
            seeds_per_fold=args.seeds_per_fold,
            base_seed=args.seed,
            output_dir=args.output,
            num_epochs=args.epochs,
            patience=args.patience,
            return_attention=args.attention,
        )
        print("CV evaluation complete!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pst",
        description="Phage–Set-Transformer utilities (train | optimise | predict | cv-evaluate)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # optimise ---------------------------------------------------------
    o = sub.add_parser("optimize", help="Hyper-parameter search with Optuna")
    _add_common_data_args(o)
    o.add_argument("-n", "--trials", type=int, default=50, help="Number of Optuna trials")
    o.add_argument("-o", "--output", default=None, help="Output directory (defaults to timestamped)")
    o.add_argument("--study-name", default=None, help="Optuna study name (optional)")
    o.add_argument("--seed", type=int, default=42, help="Random seed")
    o.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    o.add_argument("--final-seeds", type=int, default=5, help="Number of seeds for final training")
    o.add_argument("--search-config", default=None, help="Path to YAML search space config") 
    o.add_argument("--cv-epochs", type=int, default=50, help="Epochs for CV optimization (default: 50)")
    o.add_argument("--cv-patience", type=int, default=7, help="Patience for CV optimization (default: 7)")
    o.add_argument("--val-batch-size", type=int, default=None, help="Validation batch size (if None, uses same as training)")
    o.add_argument("--stability-min-epoch", type=int, default=8, help="Minimum epoch to consider for stability filtering")
    o.add_argument("--stability-loss-margin", type=float, default=0.15, help="Allowed loss increase fraction for stability check")
    o.add_argument("--stability-loss-lookback", type=int, default=8, help="Number of epochs to look back for loss comparison")
    o.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    o.set_defaults(func=_cmd_optimize)

    # train ------------------------------------------------------------
    t = sub.add_parser("train", help="Train a model with fixed parameters")
    _add_common_data_args(t)
    t.add_argument("-o", "--output", default=None, help="Output dir (defaults to timestamped)")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--patience", type=int, default=10, help="Early-stopping patience")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--normalization", choices=["none", "layer_norm", "l2_norm"], default="none", help="Input normalization type") 
    t.add_argument("--residual-classifier", action="store_true", help="Use residual connections in classifier")
    t.add_argument("--val-batch-size", type=int, default=None, help="Validation batch size (if None, uses same as training)")
    t.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    t.set_defaults(func=_cmd_train)

    # predict ----------------------------------------------------------
    p = sub.add_parser("predict", help="Run inference with a trained model")
    p.add_argument("--model", required=True, help="Path to .pt checkpoint")
    p.add_argument("--pairs", help="CSV of (strain, phage) pairs; else all-by-all")
    p.add_argument("--out", default="predictions.csv", help="Output CSV path")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--attention", action="store_true", help="Save attention weights")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    _add_common_data_args(p)
    p.set_defaults(func=_cmd_predict)

    # cv-evaluate ------------------------------------------------------
    cv = sub.add_parser("cv-evaluate", help="Cross-validation evaluation with fixed hyperparameters")
    _add_common_data_args(cv)
    cv.add_argument("--params-config", required=True, help="Path to YAML file with fixed hyperparameters")
    cv.add_argument("-o", "--output", default=None, help="Output directory (defaults to timestamped)")
    cv.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    cv.add_argument("--seeds-per-fold", type=int, default=3, help="Number of random seeds per fold")
    cv.add_argument("--epochs", type=int, default=150, help="Training epochs per model")
    cv.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    cv.add_argument("--seed", type=int, default=42, help="Base random seed")
    cv.add_argument("--attention", action="store_true", help="Save attention weights")
    cv.add_argument("--val-batch-size", type=int, default=None, help="Validation batch size")
    cv.add_argument("--job-id", type=int, default=None, help="SLURM array job ID (for single fold×seed)")
    cv.set_defaults(func=_cmd_cv_evaluate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()