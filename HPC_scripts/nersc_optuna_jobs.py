# ────────────────────────────────────────────────────────────────── CONFIG ──
CONFIG = dict(
    # ---------- data / workflow ----------
    interactions="~/data/interactions.csv",
    strain_emb="~/data/embeddings/strains", 
    phage_emb="~/data/embeddings/phages",
    output_dir="$PSCRATCH/optuna_runs",  # Use NERSC scratch
    trial_total=200,
    folds=5,
    final_seeds=0,
    search_config=None,
    # ---------- Slurm resources ----------
    account="mXXXX_g",              # Replace with your GPU project (must end in _g)
    partition="",                   # NERSC doesn't use explicit partitions
    qos="regular",                  # or "shared" for 1-2 GPUs, "debug" for testing
    time="00:30:00",               # HH:MM:SS  (per trial)
    nodes=1,
    ntasks=1,
    cpus_per_task=32,              # GPU nodes have 128 cores, allocate per GPU
    mem_per_gpu="32G",             # NERSC uses per-GPU memory allocation
    constraint="gpu",              # NERSC constraint instead of partition
    gpus_per_task=1,               # Number of GPUs per task
    conda_env="phage-set-transformer",
)

def render_slurm(idx: int, cfg: dict, array: bool) -> str:
    # Conditional search config line
    maybe_search_config = ""
    if cfg["search_config"]:
        maybe_search_config = f" \\\n    --search-config {cfg['search_config']}"
    
    return TEMPLATE.format(
        job_id=idx,
        account=cfg["account"],
        constraint=cfg["constraint"],
        qos=cfg["qos"],
        time=cfg["time"],
        nodes=cfg["nodes"],
        ntasks=cfg["ntasks"],
        cpus_per_task=cfg["cpus_per_task"],
        gpus_per_task=cfg["gpus_per_task"],
        mem_per_gpu=cfg["mem_per_gpu"],
        out_dir=cfg["output_dir"],
        array_directive="" if not array else f"#SBATCH --array=0-{cfg['trial_total']-1}%50",
        conda_env=cfg["conda_env"],
        interactions=cfg["interactions"],
        strain_emb=cfg["strain_emb"],
        phage_emb=cfg["phage_emb"],
        folds=cfg["folds"],
        final_seeds=cfg["final_seeds"],
        timestamp=cfg["timestamp"],
        seed_base=cfg["seed_base"],
        maybe_search_config=maybe_search_config,
    )

# Updated NERSC template
TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=pst_trial_{job_id}
#SBATCH --account={account}
#SBATCH --constraint={constraint}
#SBATCH --qos={qos}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-task={gpus_per_task}
#SBATCH --mem-per-gpu={mem_per_gpu}
#SBATCH --output={out_dir}/slurm-%A_%a.out
#SBATCH --error={out_dir}/slurm-%A_%a.err
{array_directive}

TRIAL_ID=${{SLURM_ARRAY_TASK_ID:-{job_id}}}

# NERSC module loading
module load python
conda activate {conda_env} 2>&1 || {{
    echo "Direct activation failed, trying with conda init..."
    conda init bash >/dev/null 2>&1
    source ~/.bashrc >/dev/null 2>&1
    conda activate {conda_env}
}}

set -euo pipefail
echo "Running trial $TRIAL_ID on $HOSTNAME at $(date)"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

export SLURM_CPU_BIND="cores"

srun python -m phage_set_transformer.cli optimize \\
    --interactions {interactions} \\
    --strain-embeddings {strain_emb} \\
    --phage-embeddings {phage_emb} \\
    --trials 1 \\
    --folds {folds} \\
    --final-seeds {final_seeds} \\
    --output {out_dir} \\
    --study-name pst_nersc_{timestamp} \\
    --seed $(( {seed_base} + $TRIAL_ID )){maybe_search_config}

echo "Finished at $(date)"
"""