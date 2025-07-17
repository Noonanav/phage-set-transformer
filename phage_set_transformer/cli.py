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
        n_folds=args.folds,                    # ADD THIS LINE
        final_seeds=args.final_seeds,          # ADD THIS LINE
        random_state=args.seed,
        study_name=args.study_name,
        output_dir=getattr(args, 'output', None),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pst",
        description="Phage–Set-Transformer utilities (train | optimise | predict)",
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

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
