"""
Visualization functions for the phage-set-transformer package.
"""

import logging
import os
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

logger = logging.getLogger(__name__)


def plot_training_history(history: Dict[str, List[float]], 
                         trial_id: str, 
                         output_dir: str) -> None:
    """
    Plot training and validation loss and AUPR over epochs.
    
    Args:
        history: Dictionary with training history
        trial_id: Identifier for this trial
        output_dir: Directory to save plots
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    epochs = len(history['train_loss'])
    plt.figure(figsize=(15, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, epochs+1), history['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.grid(alpha=0.3)

    # AUPR
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), history['train_aupr'], label='Train AUPR')
    plt.plot(range(1, epochs+1), history['val_aupr'], label='Val AUPR')
    plt.xlabel("Epoch")
    plt.ylabel("AUPR")
    plt.title("AUPR vs. Epochs")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(plots_dir, f"trial_{trial_id}_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training history plot to {plot_path}")

    # Also save the data as CSV
    history_df = pd.DataFrame({
        'epoch': list(range(1, epochs+1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_aupr': history['train_aupr'],
        'val_aupr': history['val_aupr'],
        'lr': history['lr']
    })
    csv_path = os.path.join(plots_dir, f"trial_{trial_id}_history.csv")
    history_df.to_csv(csv_path, index=False)
    logger.info(f"Saved training history data to {csv_path}")


def plot_confusion_matrix(conf_matrix: np.ndarray, 
                         trial_id: str, 
                         output_dir: str) -> None:
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix array
        trial_id: Identifier for this trial
        output_dir: Directory to save plots
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plot_path = os.path.join(plots_dir, f"trial_{trial_id}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix plot to {plot_path}")

    # Save confusion matrix data
    cm_df = pd.DataFrame(conf_matrix, 
                        columns=['Predicted Negative', 'Predicted Positive'],
                        index=['Actual Negative', 'Actual Positive'])
    csv_path = os.path.join(plots_dir, f"trial_{trial_id}_confusion_matrix.csv")
    cm_df.to_csv(csv_path)
    logger.info(f"Saved confusion matrix data to {csv_path}")


def plot_roc_curve(all_labels: np.ndarray, 
                   all_preds: np.ndarray, 
                   trial_id: str, 
                   output_dir: str) -> None:
    """
    Plot ROC curve.
    
    Args:
        all_labels: True labels
        all_preds: Predicted probabilities
        trial_id: Identifier for this trial
        output_dir: Directory to save plots
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plot_path = os.path.join(plots_dir, f"trial_{trial_id}_roc_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curve plot to {plot_path}")

    # Save ROC curve data
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    csv_path = os.path.join(plots_dir, f"trial_{trial_id}_roc_curve.csv")
    roc_df.to_csv(csv_path, index=False)
    logger.info(f"Saved ROC curve data to {csv_path}")


def plot_pr_curve(all_labels: np.ndarray, 
                  all_preds: np.ndarray, 
                  trial_id: str, 
                  output_dir: str) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        all_labels: True labels
        all_preds: Predicted probabilities
        trial_id: Identifier for this trial
        output_dir: Directory to save plots
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall_curve, precision_curve)

    plt.figure(figsize=(6, 6))
    plt.plot(recall_curve, precision_curve, color='darkorange', lw=2,
             label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    plot_path = os.path.join(plots_dir, f"trial_{trial_id}_pr_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved PR curve plot to {plot_path}")

    # Save PR curve data
    pr_df = pd.DataFrame({'recall': recall_curve, 'precision': precision_curve})
    csv_path = os.path.join(plots_dir, f"trial_{trial_id}_pr_curve.csv")
    pr_df.to_csv(csv_path, index=False)
    logger.info(f"Saved PR curve data to {csv_path}")


def visualize_optimization_history(study: Any) -> None:
    """
    Visualize the optimization history using Optuna's built-in visualizations.
    
    Args:
        study: Optuna study object
    """
    import optuna
    
    output_dir = os.environ.get('RESULTS_DIR', 'optuna_results')
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Plot optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(plots_dir, "optimization_history.png"))
        logger.info("Saved optimization history plot")

        # Plot parameter importance
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(plots_dir, "param_importances.png"))
        logger.info("Saved parameter importance plot")

        # Plot parallel coordinate
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(os.path.join(plots_dir, "parallel_coordinate.png"))
        logger.info("Saved parallel coordinate plot")

        # Plot slice
        fig = optuna.visualization.plot_slice(study)
        fig.write_image(os.path.join(plots_dir, "slice_plot.png"))
        logger.info("Saved slice plot")
    except Exception as e:
        logger.warning(f"Could not create Optuna visualization plots: {e}")
        logger.info("Install plotly and kaleido for Optuna visualizations")

    # Save study values
    study_df = study.trials_dataframe()
    csv_path = os.path.join(output_dir, "study_results.csv")
    study_df.to_csv(csv_path, index=False)
    logger.info(f"Saved study results to {csv_path}")
