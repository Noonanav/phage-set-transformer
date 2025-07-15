#!/usr/bin/env python
"""
Submit every *.slurm file in a directory to Slurm (Lawrencium).

Usage examples
--------------
# submit all .slurm files
submit_slurm_dir.py /path/to/slurm_scripts

# only those matching a pattern
submit_slurm_dir.py /path/to/slurm_scripts --pattern "trial_*.slurm"

# dry-run
submit_slurm_dir.py /path/to/slurm_scripts --dry
"""

from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path
from typing import List

def queued_jobs(user: str) -> int:
    """Return the number of running+pending jobs for the current user."""
    out = subprocess.check_output(["squeue", "-u", user, "-h"]).decode()
    return 0 if not out else len(out.strip().splitlines())

def sbatch(file: Path, dry: bool):
    cmd = ["sbatch", str(file)]
    print(" ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Submit *.slurm files in bulk")
    p.add_argument("folder", type=Path, help="Directory containing .slurm files")
    p.add_argument("--pattern", default="*.slurm",
                   help="Glob pattern (default *.slurm)")
    p.add_argument("--dry", action="store_true",
                   help="Print commands without submitting")
    p.add_argument("--limit", type=int, default=None,
                   help="Max jobs you want simultaneously queued (check every 60 s)")
    args = p.parse_args(argv)

    files = sorted(args.folder.glob(args.pattern))
    if not files:
        sys.exit(f"No files matching pattern in {args.folder}")

    user = subprocess.check_output(["whoami"]).decode().strip()
    submitted = 0
    for f in files:
        while args.limit and queued_jobs(user) >= args.limit:
            print(f"Queue full ({args.limit}). Waiting 60 s…")
            time.sleep(60)

        sbatch(f, args.dry)
        submitted += 1

    print(f"{'Would submit' if args.dry else 'Submitted'} {submitted} jobs.")

if __name__ == "__main__":
    main()
