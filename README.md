# Phage Set Transformer

A PyTorch implementation of Set Transformer architecture for predicting phage-bacteria interactions using biological language model embeddings.

## 🔬 Overview

This package provides tools for training and optimizing Set Transformer models to predict strain-level interactions between bacterial strains and phages. The model processes variable-length sets of gene embeddings for both organisms and learns to predict their interaction potential, enabling researchers to understand phage-bacteria dynamics and mediators of phage-host interactions.

### Key Features

- **Set-based Architecture**: Handles variable-length gene sets without padding inefficiencies
- **Scalable**: Works with datasets from small lab studies to large-scale microbiome surveys
- **Cross-validation Optimization**: Built-in hyperparameter search with k-fold cross-validation
- **Flexible Data Loading**: Supports multiple embedding formats (arrays, dictionaries)
- **HPC Integration**: Ready-to-use SLURM scripts for large-scale optimization
- **Comprehensive Evaluation**: Full metrics, visualizations, and attention analysis
- **Production Ready**: CLI interface and Python API for different use cases
- **Microbiome-focused**: Designed for understanding phage-bacteria dynamics in complex microbial communities

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Noonanav/phage-set-transformer.git
cd phage-set-transformer

# Install the package
pip install -e .

# Verify installation
pst --help
```

### Basic Usage

```bash
# Optimize hyperparameters with cross-validation
pst optimize \
    --interactions data/interactions.csv \
    --strain-embeddings data/embeddings/strains/ \
    --phage-embeddings data/embeddings/phages/ \
    --trials 50 \
    --output results/

# Train a model with specific parameters
pst train \
    --interactions data/interactions.csv \
    --strain-embeddings data/embeddings/strains/ \
    --phage-embeddings data/embeddings/phages/ \
    --output models/

# Make predictions
pst predict \
    --model models/model.pt \
    --strain-embeddings data/embeddings/strains/ \
    --phage-embeddings data/embeddings/phages/ \
    --pairs data/test_pairs.csv \
    --out predictions.csv
```

## 📁 Data Formats

### Interaction Data
CSV file with columns: `strain`, `phage`, `interaction`

```csv
strain,phage,interaction
strain_001,phage_A,1
strain_001,phage_B,0
strain_002,phage_A,1
```

### Embeddings

The package supports two embedding formats:

**Format 1: Simple Arrays**
```python
# strain_001.npy contains: np.array([num_genes, embedding_dim])
embeddings = np.load('strain_001.npy')  # Shape: [1234, 384]
```

**Format 2: Dictionary Format**  
```python
# strain_001.npy contains: {gene_id: embedding_vector, ...}
embeddings = np.load('strain_001.npy', allow_pickle=True).item()
# {'gene_1': array([...]), 'gene_2': array([...]), ...}
```

## 🖥️ Command Line Interface

### Optimization Command

```bash
pst optimize [OPTIONS]
```

**Required Arguments:**
- `--interactions`: Path to interactions CSV file
- `--strain-embeddings`: Directory containing strain embedding files (.npy)
- `--phage-embeddings`: Directory containing phage embedding files (.npy)

**Optional Arguments:**
- `--trials, -n`: Number of optimization trials (default: 50)
- `--folds`: Number of cross-validation folds (default: 5)
- `--final-seeds`: Number of seeds for final model training (default: 5)
- `--output, -o`: Output directory (default: timestamped directory)
- `--study-name`: Optuna study name for resumability
- `--seed`: Random seed for reproducibility (default: 42)

### Training Command

```bash
pst train [OPTIONS]
```

**Training Parameters:**
- `--epochs`: Maximum training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--batch-size`: Batch size (default: 64)
- `--patience`: Early stopping patience (default: 10)

### Prediction Command

```bash
pst predict [OPTIONS]
```

**Prediction Parameters:**
- `--model`: Path to trained model (.pt file)
- `--pairs`: CSV with strain/phage pairs (optional, defaults to all combinations)
- `--out`: Output CSV path (default: predictions.csv)
- `--batch-size`: Batch size for inference (default: 256)
- `--attention`: Save attention weights for interpretability

## 🐍 Python API

### Hyperparameter Optimization

```python
from phage_set_transformer import run_optimization

study, final_summary = run_optimization(
    interactions_path='data/interactions.csv',
    strain_embeddings_path='data/embeddings/strains/',
    phage_embeddings_path='data/embeddings/phages/',
    n_trials=100,
    n_folds=5,
    final_seeds=5,
    output_dir='results/'
)

print(f"Best MCC: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")
```

### Training with Fixed Parameters

```python
from phage_set_transformer import train_model_with_params

results = train_model_with_params(
    interactions_path='data/interactions.csv',
    strain_embeddings_path='data/embeddings/strains/',
    phage_embeddings_path='data/embeddings/phages/',
    # Model architecture
    hidden_dim=512,
    num_heads=8,
    strain_inds=256,
    phage_inds=128,
    # Training
    num_epochs=100,
    learning_rate=1e-4,
    batch_size=64,
    output_dir='models/'
)
```

### Making Predictions

```python
from phage_set_transformer import predict

predictions = predict(
    model_path='models/model.pt',
    strain_embeddings_path='data/embeddings/strains/',
    phage_embeddings_path='data/embeddings/phages/',
    interactions_path='data/test_pairs.csv',  # Optional
    return_attention=True
)

print(f"Positive predictions: {predictions['predicted_interaction'].sum()}")
```

## 🏭 HPC Usage

For large-scale hyperparameter optimization on SLURM clusters:

### Generate SLURM Scripts

```bash
# Edit configuration in prepare_optuna_jobs.py
python prepare_optuna_jobs.py --array  # Single job array
# OR
python prepare_optuna_jobs.py --split   # Individual job files
```

### Submit Jobs

```bash
# Submit all generated scripts
python submit_slurm_dir.py slurm_scripts/20241201_143022/

# With job limit
python submit_slurm_dir.py slurm_scripts/20241201_143022/ --limit 10
```

### HPC Configuration

Edit the CONFIG section in `prepare_optuna_jobs.py`:

```python
CONFIG = dict(
    # Data paths
    interactions="~/data/interactions.csv",
    strain_emb="~/data/embeddings/strains",
    phage_emb="~/data/embeddings/phages",
    
    # Resources
    account="your_account",
    partition="gpu",
    qos="normal",
    time="02:00:00",
    mem="80G",
    gres="gpu:1",
    
    # Optimization
    trial_total=200,
    folds=5,
    final_seeds=5,
)
```

## ⚙️ Model Architecture

The Set Transformer architecture is particularly well-suited for microbiome research as it naturally handles the variable-length nature of gene sets across different bacterial strains and phages. The architecture consists of:

1. **Set Encoders**: Process variable-length gene sets using Induced Set Attention Blocks (ISABs)
2. **Cross-Attention** (optional): Allow strain and phage representations to interact, modeling biological specificity
3. **Pooling**: Aggregate set representations using Pooling by Multihead Attention (PMA)
4. **Classification**: Multi-layer perceptron for final interaction prediction

This design enables the model to learn meaningful representations regardless of genome size differences, making it ideal for diverse microbiome datasets where bacterial strains and phages can vary significantly in gene content.

### Key Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `hidden_dim` | Internal representation dimension | 256-1024 |
| `num_heads` | Attention heads | 4-16 |
| `strain_inds` | Inducing points for strains | 64-512 |
| `phage_inds` | Inducing points for phages | 32-256 |
| `temperature` | Attention temperature | 0.01-1.0 |
| `dropout` | Dropout rate | 0.0-0.3 |

## 📊 Output Files

### Optimization Results
```
results/
├── study.db                 # Optuna database (resumable)
├── best_params.json         # Best hyperparameters
├── all_trials.csv          # All trial results
├── multi_seed_summary.json  # Final model statistics
└── models/
    ├── model_seed_42.pt     # Model for each seed
    ├── model_seed_43.pt
    └── ...
```

### Training Results
```
models/
├── config.json             # Training configuration
├── models/model.pt          # Trained model
├── plots/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── pr_curve.png
├── predictions.csv          # Test set predictions
└── training.log            # Training logs
```

## 🔧 Troubleshooting

### Common Issues

**Out of Memory**
- Reduce `batch_size` or `strain_inds`/`phage_inds`
- Enable chunking: set `chunk_size=64` in model config

**Poor Performance**
- Increase model capacity: `hidden_dim`, `num_heads`
- Enable cross-attention: `use_cross_attention=True` (important for capturing phage-bacteria specificity)
- Adjust class weighting: `use_phage_weights=True` (handles imbalanced interaction data common in microbiome studies)
- Consider strain-based splitting to avoid data leakage in microbiome datasets

**Imbalanced Datasets**
- Use `use_phage_weights=True` to automatically weight rare positive interactions
- Adjust `temperature` parameter (0.01-0.1) for sharper attention on important genes
- Monitor both MCC and F1 scores as they handle class imbalance better than accuracy

**Slow Training**
- Increase `batch_size` if memory allows
- Use `scheduler_type="one_cycle"` for faster convergence
- Reduce `patience` for earlier stopping

**Data Loading Errors**
- Verify embedding file formats match expectations
- Check that all strain/phage IDs in interactions have corresponding embedding files
- Ensure consistent embedding dimensions across files

### Environment Issues

**CUDA Problems**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU if needed
export CUDA_VISIBLE_DEVICES=""
```

**Dependency Conflicts**
```bash
# Clean install
pip install --force-reinstall -e .
```

## 🛠️ Development

### Setting up Development Environment

```bash
git clone https://github.com/your-username/phage-set-transformer.git
cd phage-set-transformer

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black phage_set_transformer/
flake8 phage_set_transformer/
```

### Project Structure

```
phage_set_transformer/
├── __init__.py           # Package exports
├── cli.py               # Command-line interface
├── data.py              # Data loading utilities
├── models.py            # Model architectures
├── training.py          # Training functions
├── optimization.py      # Hyperparameter search
├── evaluation.py        # Metrics and evaluation
├── visualization.py     # Plotting functions
└── utils.py            # Helper utilities
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions!

## 🐛 Issues

Report bugs and request features on [GitHub Issues](https://github.com/Noonanav/phage-set-transformer/issues).
