# Phage-Set-Transformer

Predict **strain–phage interactions** from genome-scale protein embeddings with a
Set Transformer backbone, Optuna hyper‑parameter optimisation and fully
re‑producible pipelines.

| **Docs** | **PyPI** | **Licence** | **Tests** |
|----------|----------|-------------|-----------|
| *coming soon* | *coming soon* | MIT | ![tests](https://img.shields.io/badge/build-passing-brightgreen) |

---

##  Key Features
* **Flexible Set Transformer** that can switch between self‑attention and
  cross‑attention.
* **Inducing‑point ISAB layers** to handle hundreds of genes per genome.
* **5‑fold cross‑validated Optuna search** with median MCC pruning.
* **Multi‑seed retraining** of the best hyper‑parameters for robust estimates.
* **Curated CLI** (`pst`) *and* pure‑Python API.
* **GPU‑agnostic** (PyTorch ≥ 1.13); automatic `cuda`/`cpu` selection.
* **SQLite‑backed studies** – interrupt and resume at will.

---

##  Installation
```bash
# clone & dev‑install
git clone https://github.com/your-org/phage-set-transformer.git
cd phage-set-transformer
pip install -e .

# OR: pip install from PyPI (once released)
pip install phage-set-transformer
```

<details>
  <summary><strong>Dependencies</strong></summary>

| Package | Tested version |
|---------|----------------|
| python  | 3.8 – 3.12 |
| torch   | ≥ 1.13 |
| numpy, pandas, scikit‑learn | latest |
| optuna | ≥ 3.6 |
| matplotlib, seaborn (plots) | latest |
</details>

---

##  Quick start

### 1. Prepare your data
```text
data/
├─ interactions.csv         # columns: strain, phage, interaction (0/1)
├─ embeddings/
│   ├─ strains/
│   │   ├─ strain_A.npy
│   │   └─ ...
│   └─ phages/
│       ├─ phage_001.npy
│       └─ ...
```
*Each `.npy` contains **(n_genes, 1280)** ESM‑2 embeddings.*

### 2. Hyper‑parameter search + retrain
```bash
pst optimise     --interactions data/interactions.csv     --strain-embeddings data/embeddings/strains     --phage-embeddings  data/embeddings/phages     --trials 150     --folds 5     --final-seeds 5     --output results/
```

The command will:

1. Launch / resume an Optuna study stored in `results/study.db`.
2. Evaluate **median MCC** over 5 CV folds for every trial (prunable).
3. Retrain the best hyper‑parameters with 5 different random seeds.
4. Save:
   * `results/trials/` – per‑trial metrics  
   * `results/best_params.json`  
   * `results/model_seed_*.pt` – retrained checkpoints  
   * `results/multi_seed_summary.json`

### 3. Inference
```bash
pst predict     --model results/model_seed_42.pt     --pairs unseen_pairs.csv     --strain-embeddings data/embeddings/strains     --phage-embeddings  data/embeddings/phages     --out predictions.csv
```

---

##  Python API
```python
from phage_set_transformer.optimization import run_cv_optimization
from phage_set_transformer.training import train_model_with_params
from phage_set_transformer.evaluation import evaluate_full
```

### CV optimisation
```python
study, summary = run_cv_optimization(
    interactions_path="data/interactions.csv",
    strain_embeddings_path="data/embeddings/strains",
    phage_embeddings_path="data/embeddings/phages",
    n_trials=100, n_folds=5, final_seeds=3,
)
print(study.best_params, summary["median_mcc"])
```

### Manual training
```python
result = train_model_with_params(
    interactions_path="data/interactions.csv",
    strain_embeddings_path="data/embeddings/strains",
    phage_embeddings_path="data/embeddings/phages",
    output_dir="run1/",
    num_epochs=100,
    learning_rate=3e-4,
)
print(result["metrics"]["mcc"])
```

---

##  Repository Layout
```text
phage_set_transformer/
│
├─ data.py              # loaders & Dataset
├─ models.py            # Set Transformer implementation
├─ training.py          # train / validate loops
├─ evaluation.py        # metrics & full‑set inference
├─ optimization.py      # Optuna CV search + multi‑seed retrain
├─ utils.py             # misc helpers (logging, schedulers…)
└─ cli.py               # console‑script entry point  'pst'
tests/                  # pytest unit tests (TBD)
README.md
setup.py
requirements.txt
LICENSE
```

---

##  Metrics
We report **Matthews correlation coefficient (MCC)** by default, alongside
accuracy, precision, recall, F1, ROC‑AUC & PR‑AUC.  
During optimisation the **median MCC across folds** is maximised.

---

##  Contributing
1. Fork → feature branch → PR  
2. Run `pytest -q` and `black .` before submitting  
3. For major changes open an issue first to discuss the design.

---

##  Licence
This project is licensed under the **MIT License** – see `LICENSE` for details.
