"""
Evaluation and prediction functions for the phage-set-transformer package.
"""

import logging
import os
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, 
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
from torch.utils.data import DataLoader

from .data import load_embeddings_flexible, StrainPhageDatasetWithWeights, collate_variable_sets_with_weights
from .models import FlexibleStrainPhageTransformer
from .visualization import plot_confusion_matrix, plot_roc_curve, plot_pr_curve

logger = logging.getLogger(__name__)


def evaluate_full(model: nn.Module, 
                  data_loader: DataLoader, 
                  device: torch.device, 
                  use_phage_weights: bool = True,
                  return_attention: bool = False,
                  output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of model with all metrics.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run on
        use_phage_weights: Whether to use phage-specific weights
        return_attention: Whether to save attention weights
        output_dir: Optional directory to save outputs
        
    Returns:
        Dictionary containing all metrics and predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_strains = []
    all_phages = []
    all_attention_weights = []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            # Unpack with weights
            strain_emb, phage_emb, strain_mask, phage_mask, labels, weights, strain_ids, phage_ids = batch

            strain_emb = strain_emb.to(device)
            phage_emb = phage_emb.to(device)
            strain_mask = strain_mask.to(device)
            phage_mask = phage_mask.to(device)
            labels = labels.to(device)
            weights = weights.to(device) if use_phage_weights else None

            # Forward pass
            if return_attention and model.use_cross_attention:
                logits, (strain_attn, phage_attn) = model(
                    strain_emb, phage_emb, strain_mask, phage_mask, return_attn=True
                )
                # Store attention weights
                for i in range(len(strain_ids)):
                    all_attention_weights.append({
                        'strain': strain_ids[i],
                        'phage': phage_ids[i],
                        'strain_to_phage_attn': strain_attn[i].cpu().numpy(),
                        'phage_to_strain_attn': phage_attn[i].cpu().numpy()
                    })
            else:
                logits = model(strain_emb, phage_emb, strain_mask, phage_mask)

            # Calculate loss
            if use_phage_weights:
                criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
            else:
                criterion = nn.BCEWithLogitsLoss()

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_strains.extend(strain_ids)
            all_phages.extend(phage_ids)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    loss = total_loss / len(data_loader)

    # Calculate metrics
    binary_preds = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    mcc = matthews_corrcoef(all_labels, binary_preds)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_preds)
    except:
        roc_auc = 0.5  # Default value when one class only
    
    try:
        pr_auc = average_precision_score(all_labels, all_preds)
    except:
        pr_auc = 0.5
    
    conf_matrix = confusion_matrix(all_labels, binary_preds)

    # Save outputs if directory provided
    if output_dir:
        # CREATE THE OUTPUT DIRECTORY IF IT DOESN'T EXIST
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'strain': all_strains,
            'phage': all_phages,
            'true_label': all_labels,
            'prediction': binary_preds,
            'confidence': all_preds
        })
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to {predictions_path}")

        # Save attention weights if available
        if all_attention_weights:
            attention_path = os.path.join(output_dir, 'attention_weights.npz')
            np.savez_compressed(attention_path, attention_weights=all_attention_weights)
            logger.info(f"Saved attention weights to {attention_path}")

        # Plot metrics
        plot_confusion_matrix(conf_matrix, "evaluation", output_dir)
        plot_roc_curve(all_labels, all_preds, "evaluation", output_dir)
        plot_pr_curve(all_labels, all_preds, "evaluation", output_dir)

    return {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'conf_matrix': conf_matrix,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'binary_preds': binary_preds,
        'attention_weights': all_attention_weights if return_attention else None
    }


def load_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on (if None, uses GPU if available)
        
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    elif 'best_params' in checkpoint:
        model_config = checkpoint['best_params']
    elif 'params' in checkpoint:
        model_config = checkpoint['params']
    else:
        raise ValueError("Cannot find model configuration in checkpoint")
    
    # Get embedding dimension
    embedding_dim = checkpoint.get('embedding_dim', model_config.get('embedding_dim', 384))
    
    # Initialize model
    model = FlexibleStrainPhageTransformer(
        embedding_dim=embedding_dim,
        hidden_dim=model_config.get('hidden_dim', 512),
        num_heads=model_config.get('num_heads', 8),
        strain_inds=model_config.get('strain_inds', 256),
        phage_inds=model_config.get('phage_inds', 256),
        num_isab_layers=model_config.get('num_isab_layers', 2),
        num_seeds=model_config.get('num_seeds', 1),
        dropout=model_config.get('dropout', 0.1),
        ln=model_config.get('ln', True),
        temperature=model_config.get('temperature', 0.1),
        use_cross_attention=model_config.get('use_cross_attention', True),
        classifier_hidden_layers=model_config.get('classifier_hidden_layers', 1),
        classifier_hidden_dim=model_config.get('classifier_hidden_dim', 512),
        activation_function=model_config.get('activation_function', 'gelu'),
        chunk_size=model_config.get('chunk_size', 128)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def predict(model_path: str,
            strain_embeddings_path: str,
            phage_embeddings_path: str,
            interactions_path: Optional[str] = None,
            strain_ids: Optional[List[str]] = None,
            phage_ids: Optional[List[str]] = None,
            batch_size: int = 64,
            return_attention: bool = False,
            device: Optional[torch.device] = None) -> pd.DataFrame:
    """
    Make predictions on new strain-phage pairs.
    
    Args:
        model_path: Path to trained model
        strain_embeddings_path: Path to strain embeddings directory
        phage_embeddings_path: Path to phage embeddings directory
        interactions_path: Optional CSV with strain/phage columns for specific pairs
        strain_ids: Optional list of strain IDs (if not using interactions_path)
        phage_ids: Optional list of phage IDs (if not using interactions_path)
        batch_size: Batch size for prediction
        return_attention: Whether to return attention weights
        device: Device to run on
        
    Returns:
        DataFrame with predictions and optionally attention weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    
    # Load embeddings
    logger.info("Loading embeddings...")
    strain_embeddings = load_embeddings_flexible(strain_embeddings_path)
    phage_embeddings = load_embeddings_flexible(phage_embeddings_path)
    
    # Determine pairs to predict
    if interactions_path:
        logger.info(f"Loading pairs from {interactions_path}")
        pairs_df = pd.read_csv(interactions_path)
        if 'strain' not in pairs_df.columns or 'phage' not in pairs_df.columns:
            raise ValueError("Interactions file must have 'strain' and 'phage' columns")
        
        # Filter to available embeddings
        available_strains = set(strain_embeddings.keys())
        available_phages = set(phage_embeddings.keys())
        pairs_df = pairs_df[
            pairs_df['strain'].isin(available_strains) &
            pairs_df['phage'].isin(available_phages)
        ]
        
        # Add dummy interaction column if not present
        if 'interaction' not in pairs_df.columns:
            pairs_df['interaction'] = 0
            
    else:
        # Create all pairs or specified pairs
        if strain_ids is None:
            strain_ids = list(strain_embeddings.keys())
        if phage_ids is None:
            phage_ids = list(phage_embeddings.keys())
            
        # Create all combinations
        import itertools
        pairs = list(itertools.product(strain_ids, phage_ids))
        pairs_df = pd.DataFrame(pairs, columns=['strain', 'phage'])
        pairs_df['interaction'] = 0  # Dummy value
    
    logger.info(f"Making predictions for {len(pairs_df)} pairs")
    
    # Create dataset (use empty weights dict since we're just predicting)
    dataset = StrainPhageDatasetWithWeights(
        pairs_df, strain_embeddings, phage_embeddings, {}
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_variable_sets_with_weights
    )
    
    # Make predictions
    all_preds = []
    all_strains = []
    all_phages = []
    all_attention_weights = []
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            strain_emb, phage_emb, strain_mask, phage_mask, _, _, strain_ids, phage_ids = batch
            
            strain_emb = strain_emb.to(device)
            phage_emb = phage_emb.to(device)
            strain_mask = strain_mask.to(device)
            phage_mask = phage_mask.to(device)
            
            # Forward pass
            if return_attention and model.use_cross_attention:
                logits, (strain_attn, phage_attn) = model(
                    strain_emb, phage_emb, strain_mask, phage_mask, return_attn=True
                )
                # Store attention weights
                for i in range(len(strain_ids)):
                    all_attention_weights.append({
                        'strain_to_phage_attn': strain_attn[i].cpu().numpy(),
                        'phage_to_strain_attn': phage_attn[i].cpu().numpy()
                    })
            else:
                logits = model(strain_emb, phage_emb, strain_mask, phage_mask)
            
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_strains.extend(strain_ids)
            all_phages.extend(phage_ids)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'strain': all_strains,
        'phage': all_phages,
        'interaction_probability': all_preds,
        'predicted_interaction': (np.array(all_preds) > 0.5).astype(int)
    })
    
    # Add attention weights if requested
    if return_attention and all_attention_weights:
        results_df['attention_weights'] = all_attention_weights
    
    logger.info(f"Predictions complete. Positive predictions: {results_df['predicted_interaction'].sum()}/{len(results_df)}")
    
    return results_df
