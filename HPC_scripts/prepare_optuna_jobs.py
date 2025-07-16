#!/usr/bin/env python
"""
Prepare Slurm submission scripts for running the Set-Transformer Optuna workflow
on Lawrencium.

Edit the CONFIG block below (or pass overrides on the CLI) and run:

    python prepare_optuna_jobs.py --array

to make a single job-array submission file, *or*

    python prepare_optuna_jobs.py --split

to emit one .slurm file per trial.

Then run ./submit_optuna_jobs.sh
"""

from __future__ import annotations
import argparse, json, textwrap
from pathlib import Path
from datetime import datetime

# ────────────────────────────────────────────────────────────────── CONFIG ──
CONFIG = dict(
    # ---------- data / workflow ----------
    interactions="~/data/interactions.csv",
    strain_emb="~/data/embeddings/strains",
    phage_emb="~/data/embeddings/phages",
    output_dir="~/scratch/optuna_runs",
    trial_total=200,
    folds=5,
    final_seeds=5,
    # ---------- Slurm resources ----------
    account="ac_mak",               # e.g. "es1" or "lr5"
    partition="es1",            # or lr5, lr_bigmem, gpu
    qos="es_normal",
    time="00:30:00",            # HH:MM:SS  (per trial)
    nodes=1,
    ntasks=1,
    cpus_per_task=2,
    mem="80G",
    gres="gpu:H100:1",                    # e.g. "gpu:1"  if needed
    conda_env="phage-set-transformer",  # name OR full conda activate path
)
# ────────────────────────────────────────────────────────────────────────────

TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=pst_trial_{job_id}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}{maybe_gres}
#SBATCH --output={out_dir}/slurm-%A_%a.out
#SBATCH --error={out_dir}/slurm-%A_%a.err
{array_directive}

TRIAL_ID=${{SLURM_ARRAY_TASK_ID:-{job_id}}}

module load anaconda3
conda activate {conda_env} 2>&1 || {{
    echo "Direct activation failed, trying with conda init..."
    conda init bash >/dev/null 2>&1
    source ~/.bashrc >/dev/null 2>&1
    conda activate {conda_env}
}}

set -euo pipefail
echo "Running trial $TRIAL_ID on $HOSTNAME at $(date)"

python -m phage_set_transformer.cli optimize \\
    --interactions {interactions} \\
    --strain-embeddings {strain_emb} \\
    --phage-embeddings {phage_emb} \\
    --trials 1 \\
    --folds {folds} \\
    --final-seeds {final_seeds} \\
    --output {out_dir} \\
    --study-name pst_lr_{timestamp} \\
    --seed $(( {seed_base} + $TRIAL_ID ))

echo "Finished at $(date)"
"""

def make_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def render_slurm(idx: int, cfg: dict, array: bool) -> str:
    return TEMPLATE.format(
        job_id=idx,
        account=cfg["account"],
        partition=cfg["partition"],
        qos=cfg["qos"],
        time=cfg["time"],
        nodes=cfg["nodes"],
        ntasks=cfg["ntasks"],
        cpus_per_task=cfg["cpus_per_task"],
        mem=cfg["mem"],
        maybe_gres=f"\n#SBATCH --gres={cfg['gres']}" if cfg["gres"] else "",
        out_dir=cfg["output_dir"],
        array_directive="" if not array else f"#SBATCH --array=0-{cfg['trial_total']-1}%50",
        conda_env=cfg["conda_env"],              # ← added
        interactions=cfg["interactions"],
        strain_emb=cfg["strain_emb"],
        phage_emb=cfg["phage_emb"],
        folds=cfg["folds"],
        final_seeds=cfg["final_seeds"],
        timestamp=cfg["timestamp"],
        seed_base=cfg["seed_base"],
    )

def main():
    p = argparse.ArgumentParser(description="Generate Slurm scripts for Optuna CV search")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--array", action="store_true", help="Write a single array job")
    g.add_argument("--split", action="store_true", help="Write one file per trial")
    args = p.parse_args()

    cfg = CONFIG.copy()
    cfg["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["seed_base"] = 42                       # base for reproducible per-trial seed
    cfg["output_dir"] = str(make_dir(Path(cfg["output_dir"]) / cfg["timestamp"]))

    scripts_dir = make_dir(Path("slurm_scripts") / cfg["timestamp"])

    if args.array:
        script = render_slurm(0, cfg, array=True)
        (scripts_dir / "optuna_array.slurm").write_text(script)
        print(f"Wrote job array file: {scripts_dir/'optuna_array.slurm'}")

    else:  # split
        for i in range(cfg["trial_total"]):
            s = render_slurm(i, cfg, array=False)
            (scripts_dir / f"trial_{i}.slurm").write_text(s)
        print(f"Wrote {cfg['trial_total']} individual .slurm files to {scripts_dir}")

if __name__ == "__main__":
    main()
