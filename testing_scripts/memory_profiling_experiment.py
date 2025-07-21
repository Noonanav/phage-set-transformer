#!/usr/bin/env python
"""
Memory profiling experiment for phage-set-transformer.

This script tests the maximum batch size possible with the most complex model
architecture and largest strain data before hitting GPU memory limits.

Usage:
    python memory_profiling_experiment.py --output-dir results/memory_profile
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add the package to Python path if needed  
import importlib.util
if importlib.util.find_spec('phage_set_transformer') is None:
    sys.path.insert(0, '.')
    sys.path.insert(0, '..')

try:
    from phage_set_transformer.models import FlexibleStrainPhageTransformer, init_attention_weights
    from phage_set_transformer.data import collate_variable_sets_with_weights
    from phage_set_transformer.utils import setup_logging, get_device
except ImportError as e:
    print(f"Error importing phage_set_transformer: {e}")
    print("Make sure you're running from the package directory or have installed the package.")
    sys.exit(1)

logger = logging.getLogger(__name__)


class SyntheticLargeStrainDataset(Dataset):
    """Dataset with synthetic large strain data for memory testing."""
    
    def __init__(self, 
                 num_samples: int,
                 embedding_dim: int = 1052,
                 strain_sizes: List[int] = None,
                 phage_sizes: List[int] = None):
        """
        Create synthetic dataset with specified characteristics.
        
        Args:
            num_samples: Number of strain-phage pairs
            embedding_dim: Dimension of embeddings (1052 for your case)
            strain_sizes: List of strain sizes (number of vectors per strain)
            phage_sizes: List of phage sizes (number of vectors per phage)
        """
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        
        # Default to large strain sizes if not specified
        if strain_sizes is None:
            strain_sizes = [3000, 3500, 4000, 4500, 5000]  # Large strains
        if phage_sizes is None:
            phage_sizes = [100, 200, 300, 400, 500]  # Moderate phage sizes
            
        self.strain_sizes = strain_sizes
        self.phage_sizes = phage_sizes
        
        # Pre-generate strain and phage sizes for each sample
        self.sample_strain_sizes = np.random.choice(strain_sizes, num_samples)
        self.sample_phage_sizes = np.random.choice(phage_sizes, num_samples)
        
        # Generate labels and weights
        self.labels = np.random.randint(0, 2, num_samples)
        self.weights = np.ones(num_samples)  # Uniform weights for simplicity
        
        logger.info(f"Created synthetic dataset with {num_samples} samples")
        logger.info(f"Embedding dim: {embedding_dim}")
        logger.info(f"Strain sizes: {min(strain_sizes)}-{max(strain_sizes)}")
        logger.info(f"Phage sizes: {min(phage_sizes)}-{max(phage_sizes)}")
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        strain_size = self.sample_strain_sizes[idx]
        phage_size = self.sample_phage_sizes[idx]
        
        # Generate random embeddings
        strain_emb = torch.randn(strain_size, self.embedding_dim, dtype=torch.float32)
        phage_emb = torch.randn(phage_size, self.embedding_dim, dtype=torch.float32)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        
        strain_id = f"strain_{idx}"
        phage_id = f"phage_{idx}"
        
        return strain_emb, phage_emb, label, weight, strain_id, phage_id


def get_max_complexity_config(embedding_dim: int = 1052) -> Dict[str, Any]:
    """
    Get the most complex model configuration for memory testing.
    
    Args:
        embedding_dim: Input embedding dimension
        
    Returns:
        Model configuration dictionary
    """
    return {
        'embedding_dim': embedding_dim,
        'hidden_dim': 1024,  # Large hidden dimension
        'num_heads': 16,  # Many attention heads
        'strain_inds': 512,  # Many inducing points for strains
        'phage_inds': 256,  # Many inducing points for phages
        'num_isab_layers': 4,  # Deep ISAB layers
        'num_seeds': 8,  # Multiple seeds for PMA
        'dropout': 0.1,
        'ln': True,  # Layer normalization
        'temperature': 0.1,
        'use_cross_attention': True,  # Cross-attention is memory intensive
        'classifier_hidden_layers': 6,  # Deep classifier
        'classifier_hidden_dim': 1024,  # Large classifier hidden dim
        'activation_function': 'gelu',
        'chunk_size': 64,  # Smaller chunks for large sets
        'normalization_type': 'layer_norm',
        'use_residual_classifier': True,  # Residual connections
    }


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and initialize the model."""
    model = FlexibleStrainPhageTransformer(**config)
    model = init_attention_weights(model)
    model = model.to(device)
    return model


def measure_memory_usage(model: nn.Module, 
                        data_loader: DataLoader, 
                        device: torch.device,
                        forward_only: bool = True) -> Dict[str, float]:
    """
    Measure GPU memory usage for a given batch size.
    
    Args:
        model: Model to test
        data_loader: Data loader with specific batch size
        device: GPU device
        forward_only: If True, only do forward pass. If False, include backward pass.
        
    Returns:
        Dictionary with memory statistics in GB
    """
    model.eval() if forward_only else model.train()
    
    # Clear cache and reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    memory_stats = {}
    
    try:
        # Get one batch
        batch = next(iter(data_loader))
        strain_emb, phage_emb, strain_mask, phage_mask, labels, weights = batch[:6]
        
        # Move to device
        strain_emb = strain_emb.to(device, non_blocking=True)
        phage_emb = phage_emb.to(device, non_blocking=True)
        strain_mask = strain_mask.to(device, non_blocking=True)
        phage_mask = phage_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        
        # Measure memory after data loading
        torch.cuda.synchronize(device)  # Ensure all operations complete
        memory_stats['data_loaded_gb'] = torch.cuda.memory_allocated(device) / 1e9
        
        # Forward pass
        if forward_only:
            with torch.no_grad():
                logits = model(strain_emb, phage_emb, strain_mask, phage_mask)
        else:
            logits = model(strain_emb, phage_emb, strain_mask, phage_mask)
        
        # Measure memory after forward pass
        torch.cuda.synchronize(device)
        memory_stats['forward_pass_gb'] = torch.cuda.memory_allocated(device) / 1e9
        memory_stats['peak_forward_gb'] = torch.cuda.max_memory_allocated(device) / 1e9
        
        if not forward_only:
            # Backward pass
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, labels)
            loss.backward()
            
            # Measure memory after backward pass
            torch.cuda.synchronize(device)
            memory_stats['backward_pass_gb'] = torch.cuda.memory_allocated(device) / 1e9
            memory_stats['peak_total_gb'] = torch.cuda.max_memory_allocated(device) / 1e9
        
        memory_stats['success'] = True
        
    except torch.cuda.OutOfMemoryError as e:
        memory_stats['success'] = False
        memory_stats['error'] = str(e)
        memory_stats['peak_total_gb'] = torch.cuda.max_memory_allocated(device) / 1e9
        
    except Exception as e:
        memory_stats['success'] = False
        memory_stats['error'] = f"Unexpected error: {str(e)}"
        memory_stats['peak_total_gb'] = torch.cuda.max_memory_allocated(device) / 1e9
    
    finally:
        # Clean up - be thorough
        if 'batch' in locals():
            del batch
        if 'strain_emb' in locals():
            del strain_emb, phage_emb, strain_mask, phage_mask, labels, weights
        if 'logits' in locals():
            del logits
        if 'loss' in locals():
            del loss
        torch.cuda.empty_cache()
        gc.collect()
    
    return memory_stats


def run_batch_size_experiment(output_dir: str,
                             embedding_dim: int = 1052,
                             max_batch_size: int = 128,
                             num_samples: int = 1000,
                             include_backward: bool = False) -> pd.DataFrame:
    """
    Run the main batch size vs memory experiment.
    
    Args:
        output_dir: Directory to save results
        embedding_dim: Input embedding dimension
        max_batch_size: Maximum batch size to test
        num_samples: Number of samples in synthetic dataset
        include_backward: Whether to test backward pass as well
        
    Returns:
        DataFrame with results
    """
    device = get_device()
    if device.type != 'cuda':
        raise RuntimeError("This experiment requires a CUDA GPU")
    
    logger.info(f"Running memory profiling on {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model with max complexity
    model_config = get_max_complexity_config(embedding_dim)
    logger.info("Model configuration:")
    for key, value in model_config.items():
        logger.info(f"  {key}: {value}")
    
    model = create_model(model_config, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create synthetic dataset - smaller since we only need one batch per test
    dataset = SyntheticLargeStrainDataset(
        num_samples=max_batch_size * 2,  # Only need 2x max batch size samples
        embedding_dim=embedding_dim,
        strain_sizes=[3000, 3500, 4000, 4500, 5000],  # Large strains
        phage_sizes=[100, 200, 300, 400, 500]  # Moderate phages
    )
    
    # Test different batch sizes
    results = []
    batch_sizes = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128]
    batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
    
    logger.info(f"Testing batch sizes: {batch_sizes}")
    
    for batch_size in batch_sizes:
        logger.info(f"\n--- Testing batch size: {batch_size} ---")
        
        # Create data loader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,  # We'll handle GPU transfer manually
            collate_fn=collate_variable_sets_with_weights
        )
        
        # Test forward pass
        logger.info("Testing forward pass...")
        forward_stats = measure_memory_usage(model, data_loader, device, forward_only=True)
        
        result = {
            'batch_size': batch_size,
            'forward_success': forward_stats['success'],
            'forward_peak_gb': forward_stats.get('peak_forward_gb', 0),
        }
        
        if forward_stats['success']:
            logger.info(f"  Forward pass successful - Peak memory: {forward_stats['peak_forward_gb']:.2f} GB")
            
            if include_backward:
                # Test backward pass
                logger.info("Testing backward pass...")
                backward_stats = measure_memory_usage(model, data_loader, device, forward_only=False)
                result.update({
                    'backward_success': backward_stats['success'],
                    'backward_peak_gb': backward_stats.get('peak_total_gb', 0),
                })
                
                if backward_stats['success']:
                    logger.info(f"  Backward pass successful - Peak memory: {backward_stats['peak_total_gb']:.2f} GB")
                else:
                    logger.info(f"  Backward pass failed: {backward_stats.get('error', 'Unknown error')}")
        else:
            logger.info(f"  Forward pass failed: {forward_stats.get('error', 'Unknown error')}")
            result.update({
                'backward_success': False,
                'backward_peak_gb': 0,
            })
        
        results.append(result)
        
        # Stop if we hit OOM on forward pass
        if not forward_stats['success']:
            logger.info("Stopping experiment due to forward pass OOM")
            break
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'memory_profiling_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Save model config
    config_path = os.path.join(output_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    return results_df


def plot_memory_results(results_df: pd.DataFrame, output_dir: str, include_backward: bool = False):
    """Create plots of memory usage vs batch size."""
    
    plt.style.use('default')
    
    if include_backward:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Forward pass plot
    successful_forward = results_df[results_df['forward_success'] == True]
    failed_forward = results_df[results_df['forward_success'] == False]
    
    if len(successful_forward) > 0:
        ax1.plot(successful_forward['batch_size'], successful_forward['forward_peak_gb'], 
                'o-', linewidth=2, markersize=8, label='Successful', color='green')
    
    if len(failed_forward) > 0:
        ax1.scatter(failed_forward['batch_size'], [80] * len(failed_forward), 
                   color='red', s=100, marker='x', label='OOM', zorder=5)
    
    ax1.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='H100 Memory Limit (80GB)')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Peak GPU Memory (GB)')
    ax1.set_title('Forward Pass Memory Usage vs Batch Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 85)
    
    if include_backward:
        # Backward pass plot
        successful_backward = results_df[results_df['backward_success'] == True]
        failed_backward = results_df[results_df['backward_success'] == False]
        
        if len(successful_backward) > 0:
            ax2.plot(successful_backward['batch_size'], successful_backward['backward_peak_gb'], 
                    'o-', linewidth=2, markersize=8, label='Successful', color='green')
        
        if len(failed_backward) > 0:
            failed_batch_sizes = failed_backward['batch_size'].tolist()
            ax2.scatter(failed_batch_sizes, [80] * len(failed_batch_sizes), 
                       color='red', s=100, marker='x', label='OOM', zorder=5)
        
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='H100 Memory Limit (80GB)')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Peak GPU Memory (GB)')
        ax2.set_title('Forward + Backward Pass Memory Usage vs Batch Size')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 85)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'memory_usage_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot to {plot_path}")
    
    # Create summary statistics
    summary = {}
    if len(successful_forward) > 0:
        max_forward_batch = successful_forward['batch_size'].max()
        summary['max_forward_batch_size'] = int(max_forward_batch)
        summary['max_forward_memory_gb'] = float(successful_forward['forward_peak_gb'].max())
    
    if include_backward and len(successful_backward) > 0:
        max_backward_batch = successful_backward['batch_size'].max()
        summary['max_backward_batch_size'] = int(max_backward_batch)
        summary['max_backward_memory_gb'] = float(successful_backward['backward_peak_gb'].max())
    
    summary_path = os.path.join(output_dir, 'memory_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Memory profiling summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Memory profiling experiment for phage-set-transformer")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--embedding-dim", type=int, default=1052, help="Embedding dimension")
    parser.add_argument("--max-batch-size", type=int, default=128, help="Maximum batch size to test")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples in dataset")
    parser.add_argument("--include-backward", action="store_true", help="Test backward pass as well")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, "memory_profiling.log")
    setup_logging(level=args.log_level, log_file=log_file)
    
    logger.info("="*80)
    logger.info("MEMORY PROFILING EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Embedding dimension: {args.embedding_dim}")
    logger.info(f"Max batch size: {args.max_batch_size}")
    logger.info(f"Include backward pass: {args.include_backward}")
    
    try:
        # Run experiment
        results_df = run_batch_size_experiment(
            output_dir=args.output_dir,
            embedding_dim=args.embedding_dim,
            max_batch_size=args.max_batch_size,
            num_samples=args.num_samples,
            include_backward=args.include_backward
        )
        
        # Create plots
        plot_memory_results(results_df, args.output_dir, args.include_backward)
        
        logger.info("="*80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
