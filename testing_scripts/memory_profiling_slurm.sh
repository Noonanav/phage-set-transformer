#!/bin/bash
#SBATCH --job-name=pst_memory_profile
#SBATCH --account=ac_mak
#SBATCH --partition=es1
#SBATCH --qos=es_normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:H100:1
#SBATCH --output=memory_profile-%j.out
#SBATCH --error=memory_profile-%j.err

# Load modules
module load anaconda3

# Activate conda environment
conda activate phage-set-transformer 2>&1 || {
    echo "Direct activation failed, trying with conda init..."
    conda init bash >/dev/null 2>&1
    source ~/.bashrc >/dev/null 2>&1
    conda activate phage-set-transformer
}

# Set up error handling
set -euo pipefail

echo "==============================================="
echo "Memory Profiling Experiment Started"
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MiB"
echo "==============================================="

# Navigate to the directory containing the scripts
cd "$SLURM_SUBMIT_DIR" || {
    echo "Error: Could not change to submit directory"
    exit 1
}

echo "Working directory: $(pwd)"
echo "Python path: $(which python)"

# Verify the experiment script exists
if [[ ! -f "memory_profiling_experiment.py" ]]; then
    echo "Error: memory_profiling_experiment.py not found in $(pwd)"
    echo "Available files:"
    ls -la *.py 2>/dev/null || echo "No Python files found"
    exit 1
fi

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$HOME/scratch/memory_profiling_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"

# Test GPU availability
echo "Testing GPU availability..."
nvidia-smi || {
    echo "Error: GPU not available"
    exit 1
}

# Run the memory profiling experiment
echo "Starting memory profiling..."

python memory_profiling_experiment.py \
    --output-dir "$OUTPUT_DIR" \
    --embedding-dim 1052 \
    --max-batch-size 128 \
    --num-samples 1000 \
    --include-backward \
    --log-level INFO

echo "==============================================="
echo "Memory Profiling Experiment Completed"
echo "==============================================="
echo "Results saved to: $OUTPUT_DIR"
echo "Date: $(date)"

# Display final GPU memory status
echo "Final GPU memory status:"
nvidia-smi

# Copy important results to a easy-to-find location
RESULTS_SUMMARY="$HOME/memory_profiling_latest"
cp -r "$OUTPUT_DIR" "$RESULTS_SUMMARY" 2>/dev/null || echo "Could not copy to summary location"
echo "Results also copied to: $RESULTS_SUMMARY"
