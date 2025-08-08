#!/usr/bin/env python
"""
Prepare Slurm submission scripts for CV evaluation with fixed hyperparameters.

Usage:
    python prepare_cv_evaluation_jobs.py --params-config best_params.yaml
"""

import argparse
from pathlib import Path
from datetime import datetime

# ────────────────────────────────────────────────────────────────── CONFIG ──
CONFIG = dict(
    # data
    interactions="~/data/interactions.csv",
    strain_emb="~/data/embeddings/strains", 
    phage_emb="~/data/embeddings/phages",
    output_dir="~/scratch/cv_evaluation_runs",
    folds=5,
    seeds_per_fold=3,
    epochs=150,
    patience=15,
    base_seed=42,
    # slurm
    account="ac_mak",
    partition="es1",
    qos="es_normal", 
    time="02:00:00",
    nodes=1,
    ntasks=1,
    cpus_per_task=4,
    mem="80G",
    gres="gpu:H100:1",
    conda_env="phage-set-transformer",
)

TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=pst_cv_{timestamp}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}{maybe_gres}
#SBATCH --array=0-{max_job_id}%{max_concurrent}
#SBATCH --output={out_dir}/slurm-%A_%a.out
#SBATCH --error={out_dir}/slurm-%A_%a.err

module load anaconda3
conda activate {conda_env}

set -euo pipefail
echo "Running CV job $SLURM_ARRAY_TASK_ID on $HOSTNAME at $(date)"

python -m phage_set_transformer.cli cv-evaluate \\
    --interactions {interactions} \\
    --strain-embeddings {strain_emb} \\
    --phage-embeddings {phage_emb} \\
    --params-config {params_config} \\
    --folds {folds} \\
    --seeds-per-fold {seeds_per_fold} \\
    --epochs {epochs} \\
    --patience {patience} \\
    --seed {base_seed} \\
    --output {out_dir} \\
    --attention \\
    --job-id $SLURM_ARRAY_TASK_ID

echo "Completed at $(date)"
"""

def main():
    p = argparse.ArgumentParser(description="Generate CV evaluation SLURM scripts")
    p.add_argument("--params-config", required=True,
                   help="Path to YAML config with fixed hyperparameters")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--seeds-per-fold", type=int, default=None)
    p.add_argument("--max-concurrent", type=int, default=25)
    
    args = p.parse_args()

    cfg = CONFIG.copy()
    cfg["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["params_config"] = args.params_config
    
    if args.folds:
        cfg["folds"] = args.folds
    if args.seeds_per_fold:
        cfg["seeds_per_fold"] = args.seeds_per_fold
        
    # Create directories
    output_dir = Path(cfg["output_dir"]) / cfg["timestamp"]
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)
    
    scripts_dir = Path("cv_slurm_scripts") / cfg["timestamp"]
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate array parameters
    total_jobs = cfg["folds"] * cfg["seeds_per_fold"]
    max_job_id = total_jobs - 1
    
    # Generate script
    script = TEMPLATE.format(
        timestamp=cfg["timestamp"],
        account=cfg["account"],
        partition=cfg["partition"],
        qos=cfg["qos"],
        time=cfg["time"],
        nodes=cfg["nodes"],
        ntasks=cfg["ntasks"],
        cpus_per_task=cfg["cpus_per_task"],
        mem=cfg["mem"],
        maybe_gres=f"\n#SBATCH --gres={cfg['gres']}" if cfg["gres"] else "",
        max_job_id=max_job_id,
        max_concurrent=args.max_concurrent,
        out_dir=cfg["output_dir"],
        folds=cfg["folds"],
        seeds_per_fold=cfg["seeds_per_fold"],
        conda_env=cfg["conda_env"],
        interactions=cfg["interactions"],
        strain_emb=cfg["strain_emb"],
        phage_emb=cfg["phage_emb"],
        params_config=cfg["params_config"],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        base_seed=cfg["base_seed"],
    )
    
    script_path = scripts_dir / "cv_evaluation.slurm"
    script_path.write_text(script)
    
    # Generate submission script
    submit_script = f"""#!/bin/bash
echo "Submitting CV evaluation: {total_jobs} jobs ({cfg['folds']} folds × {cfg['seeds_per_fold']} seeds)"
sbatch {script_path}
echo "Monitor: watch 'squeue -u $USER'"
echo "Results: {cfg['output_dir']}"
"""
    
    submit_path = scripts_dir / "submit_cv.sh"
    submit_path.write_text(submit_script)
    submit_path.chmod(0o755)
    
    print(f"Generated CV evaluation scripts:")
    print(f"  Jobs: {total_jobs} ({cfg['folds']} folds × {cfg['seeds_per_fold']} seeds)")
    print(f"  Script: {script_path}")
    print(f"  Submit: {submit_path}")
    print(f"  Results: {cfg['output_dir']}")

if __name__ == "__main__":
    main()
