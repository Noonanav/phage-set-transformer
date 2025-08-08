#!/usr/bin/env python
"""
Simple aggregation of CV evaluation results.

Usage:
    python aggregate_cv_results.py --results-dir cv_results --folds 5 --seeds-per-fold 3
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def find_completed_runs(results_dir: Path) -> List[Dict[str, Any]]:
    """Find all completed CV runs."""
    runs = []
    
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith('fold_'):
            continue
            
        # Check for required files
        summary_path = run_dir / "summary.json"
        model_path = run_dir / "model.pt"
        predictions_path = run_dir / "predictions.csv"
        
        if not all(p.exists() for p in [summary_path, model_path]):
            logger.warning(f"Incomplete run: {run_dir}")
            continue
            
        # Load summary
        with open(summary_path, 'r') as f:
            run_info = json.load(f)
        
        run_info['run_dir'] = str(run_dir)
        run_info['model_path'] = str(model_path)
        if predictions_path.exists():
            run_info['predictions_path'] = str(predictions_path)
        
        runs.append(run_info)
    
    return sorted(runs, key=lambda x: (x['fold_id'], x['seed']))


def aggregate_metrics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute basic statistics across all runs."""
    if not runs:
        return {}
    
    # Collect MCC values
    mccs = [run['final_mcc'] for run in runs]
    
    # Group by fold
    by_fold = {}
    for run in runs:
        fold_id = run['fold_id']
        if fold_id not in by_fold:
            by_fold[fold_id] = []
        by_fold[fold_id].append(run['final_mcc'])
    
    # Compute statistics
    stats = {
        'overall': {
            'n_runs': len(runs),
            'mcc_mean': float(np.mean(mccs)),
            'mcc_std': float(np.std(mccs)),
            'mcc_min': float(np.min(mccs)),
            'mcc_max': float(np.max(mccs)),
            'mcc_median': float(np.median(mccs)),
        },
        'by_fold': {}
    }
    
    for fold_id, fold_mccs in by_fold.items():
        stats['by_fold'][fold_id] = {
            'n_runs': len(fold_mccs),
            'mcc_mean': float(np.mean(fold_mccs)),
            'mcc_std': float(np.std(fold_mccs)),
            'mcc_values': fold_mccs,
        }
    
    return stats


def create_ensemble_predictions(runs: List[Dict[str, Any]], output_dir: Path) -> None:
    """Create ensemble predictions if prediction files exist."""
    pred_files = [run for run in runs if 'predictions_path' in run]
    
    if not pred_files:
        logger.info("No prediction files found, skipping ensemble predictions")
        return
    
    logger.info(f"Creating ensemble from {len(pred_files)} prediction files")
    
    # Load all predictions
    all_preds = []
    for run in pred_files:
        df = pd.read_csv(run['predictions_path'])
        df['fold_id'] = run['fold_id']
        df['seed'] = run['seed']
        all_preds.append(df)
    
    if not all_preds:
        return
    
    # Combine predictions
    combined_df = pd.concat(all_preds, ignore_index=True)
    
    # Create ensemble by averaging
    ensemble_df = combined_df.groupby(['strain', 'phage']).agg({
        'true_label': 'first',
        'confidence': ['mean', 'std', 'count'],
    }).round(4)
    
    # Flatten columns
    ensemble_df.columns = ['true_label', 'confidence_mean', 'confidence_std', 'n_models']
    ensemble_df = ensemble_df.reset_index()
    ensemble_df['ensemble_prediction'] = (ensemble_df['confidence_mean'] > 0.5).astype(int)
    
    # Save ensemble predictions
    ensemble_path = output_dir / "ensemble_predictions.csv"
    ensemble_df.to_csv(ensemble_path, index=False)
    logger.info(f"Saved ensemble predictions to {ensemble_path}")
    
    # Save individual predictions
    individual_path = output_dir / "all_predictions.csv"
    combined_df.to_csv(individual_path, index=False)
    logger.info(f"Saved individual predictions to {individual_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate CV evaluation results")
    parser.add_argument("--results-dir", required=True, type=Path,
                       help="Directory containing CV results")
    parser.add_argument("--folds", type=int, required=True,
                       help="Expected number of folds")
    parser.add_argument("--seeds-per-fold", type=int, required=True,
                       help="Expected number of seeds per fold")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.results_dir / "aggregated_results"
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Aggregating results from {args.results_dir}")
    expected_runs = args.folds * args.seeds_per_fold
    
    # Find completed runs
    runs = find_completed_runs(args.results_dir)
    logger.info(f"Found {len(runs)} completed runs (expected {expected_runs})")
    
    if not runs:
        logger.error("No completed runs found!")
        return
    
    # Aggregate metrics
    stats = aggregate_metrics(runs)
    
    # Create ensemble predictions
    create_ensemble_predictions(runs, output_dir)
    
    # Save summary
    summary = {
        'cv_stats': stats,
        'runs': runs,
        'expected_runs': expected_runs,
        'completed_runs': len(runs),
        'completion_rate': len(runs) / expected_runs,
    }
    
    summary_path = output_dir / "cv_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    # Print results
    overall = stats['overall']
    print(f"\nCV Evaluation Results:")
    print(f"Completed runs: {len(runs)}/{expected_runs}")
    print(f"Overall MCC: {overall['mcc_mean']:.4f} ± {overall['mcc_std']:.4f}")
    print(f"MCC range: [{overall['mcc_min']:.4f}, {overall['mcc_max']:.4f}]")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
