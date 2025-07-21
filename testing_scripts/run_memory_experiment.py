#!/usr/bin/env python
"""
Easy script to run and analyze memory profiling experiments.

Usage:
    # Run locally (if you have GPU access)
    python run_memory_experiment.py --local

    # Submit to SLURM
    python run_memory_experiment.py --submit

    # Analyze existing results
    python run_memory_experiment.py --analyze /path/to/results
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


def submit_slurm_job():
    """Submit the memory profiling job to SLURM."""
    script_path = "memory_profiling.slurm"
    
    # Check if we're in the right directory
    required_files = ["memory_profiling.slurm", "memory_profiling_experiment.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Required files not found: {missing_files}")
        print(f"Current directory: {os.getcwd()}")
        print("Make sure you're running from the directory containing:")
        for f in required_files:
            print(f"  - {f}")
        return False
    
    try:
        # Test if sbatch is available
        subprocess.run(["sbatch", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: sbatch command not found. Are you on a SLURM cluster?")
        return False
    
    try:
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"✓ Job submitted successfully!")
            print(f"Job ID: {job_id}")
            print(f"Monitor with: squeue -u $USER")
            print(f"View output with: tail -f memory_profile-{job_id}.out")
            print(f"Results will be saved to: ~/scratch/memory_profiling_<timestamp>")
            return True
        else:
            print(f"Error submitting job: {result.stderr}")
            return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def run_local_experiment():
    """Run the experiment locally."""
    print("Running memory profiling experiment locally...")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"memory_profiling_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the experiment
    cmd = [
        "python", "memory_profiling_experiment.py",
        "--output-dir", output_dir,
        "--embedding-dim", "1052",
        "--max-batch-size", "128",
        "--num-samples", "1000",
        "--include-backward",
        "--log-level", "INFO"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Experiment completed successfully!")
        print(f"Results saved to: {output_dir}")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return None


def analyze_results(results_dir: str):
    """Analyze and display results from a completed experiment."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found!")
        return
    
    # Load results
    results_file = results_dir / "memory_profiling_results.csv"
    config_file = results_dir / "model_config.json"
    summary_file = results_dir / "memory_summary.json"
    
    if not results_file.exists():
        print(f"Error: Results file {results_file} not found!")
        return
    
    # Load data
    results_df = pd.read_csv(results_file)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}
    
    # Display summary
    print("="*60)
    print("MEMORY PROFILING RESULTS SUMMARY")
    print("="*60)
    
    if config:
        print("\nModel Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print(f"\nDataset Characteristics:")
    print(f"  Embedding dimension: {config.get('embedding_dim', 'Unknown')}")
    print(f"  Max strain size: 5000 vectors")
    print(f"  Max phage size: 500 vectors")
    
    print(f"\nBatch Size Analysis:")
    print(f"  Total batch sizes tested: {len(results_df)}")
    print(f"  Successful forward passes: {results_df['forward_success'].sum()}")
    
    if 'backward_success' in results_df.columns:
        print(f"  Successful backward passes: {results_df['backward_success'].sum()}")
    
    if summary:
        print(f"\nMemory Limits Found:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    # Show detailed results table
    print(f"\nDetailed Results:")
    print("-" * 80)
    
    display_df = results_df.copy()
    display_df['forward_peak_gb'] = display_df['forward_peak_gb'].round(2)
    if 'backward_peak_gb' in display_df.columns:
        display_df['backward_peak_gb'] = display_df['backward_peak_gb'].round(2)
    
    print(display_df.to_string(index=False))
    
    # Show recommendations
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    successful_forward = results_df[results_df['forward_success'] == True]
    if len(successful_forward) > 0:
        max_forward_batch = successful_forward['batch_size'].max()
        max_forward_memory = successful_forward['forward_peak_gb'].max()
        
        print(f"Maximum safe batch size for inference: {max_forward_batch}")
        print(f"  (Peak memory usage: {max_forward_memory:.2f} GB)")
        
        # Conservative recommendation
        conservative_batch = max(1, int(max_forward_batch * 0.8))
        print(f"Conservative recommendation for inference: {conservative_batch}")
        print(f"  (80% of maximum for safety margin)")
    
    if 'backward_success' in results_df.columns:
        successful_backward = results_df[results_df['backward_success'] == True]
        if len(successful_backward) > 0:
            max_backward_batch = successful_backward['batch_size'].max()
            max_backward_memory = successful_backward['backward_peak_gb'].max()
            
            print(f"\nMaximum safe batch size for training: {max_backward_batch}")
            print(f"  (Peak memory usage: {max_backward_memory:.2f} GB)")
            
            # Conservative recommendation
            conservative_train_batch = max(1, int(max_backward_batch * 0.8))
            print(f"Conservative recommendation for training: {conservative_train_batch}")
            print(f"  (80% of maximum for safety margin)")
    
    print(f"\nNote: These recommendations are for the most complex model architecture.")
    print(f"You may be able to use larger batch sizes with simpler configurations.")
    
    # Check if plot exists
    plot_file = results_dir / "memory_usage_plot.png"
    if plot_file.exists():
        print(f"\nMemory usage plot saved to: {plot_file}")
    
    print(f"\nFull results available in: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Memory profiling experiment runner and analyzer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submit", action="store_true", help="Submit job to SLURM")
    group.add_argument("--local", action="store_true", help="Run experiment locally")
    group.add_argument("--analyze", metavar="DIR", help="Analyze results from directory")
    
    args = parser.parse_args()
    
    if args.submit:
        submit_slurm_job()
    elif args.local:
        output_dir = run_local_experiment()
        if output_dir:
            print(f"\nAnalyzing results...")
            analyze_results(output_dir)
    elif args.analyze:
        analyze_results(args.analyze)


if __name__ == "__main__":
    main()
