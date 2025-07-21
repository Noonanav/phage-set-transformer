#!/usr/bin/env python
"""
Setup verification script for memory profiling experiment.

This script checks that all requirements are met before running the experiment.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path


def check_files():
    """Check that required files exist."""
    print("Checking required files...")
    
    required_files = [
        "memory_profiling_experiment.py",
        "memory_profiling.slurm", 
        "run_memory_experiment.py"
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file}")
            missing.append(file)
    
    return len(missing) == 0, missing


def check_python_packages():
    """Check that required Python packages are available."""
    print("\nChecking Python packages...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "scikit-learn")
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name}")
                missing.append(name)
        except ImportError:
            print(f"  ✗ {name}")
            missing.append(name)
    
    return len(missing) == 0, missing


def check_phage_transformer_package():
    """Check if phage-set-transformer package is available."""
    print("\nChecking phage-set-transformer package...")
    
    # Try different import strategies
    try:
        import phage_set_transformer
        print("  ✓ Package installed and importable")
        return True, None
    except ImportError:
        pass
    
    # Try with current directory in path
    if os.path.exists("phage_set_transformer") or os.path.exists("__init__.py"):
        print("  ✓ Package found in current directory")
        return True, None
    
    # Check parent directory
    parent_files = list(Path("..").glob("**/phage_set_transformer"))
    if parent_files:
        print(f"  ✓ Package found in parent directory: {parent_files[0]}")
        return True, None
    
    print("  ✗ phage-set-transformer package not found")
    return False, "Package not installed or not in Python path"


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  ✓ CUDA available with {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                print(f"    GPU {i}: {name} ({memory_gb:.1f} GB)")
            
            return True, None
        else:
            print("  ✗ CUDA not available")
            return False, "CUDA not available"
    except ImportError:
        print("  ✗ PyTorch not available")
        return False, "PyTorch not installed"


def check_slurm():
    """Check if SLURM is available."""
    print("\nChecking SLURM availability...")
    
    try:
        result = subprocess.run(["sbatch", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✓ SLURM available: {version}")
            return True, None
        else:
            print("  ✗ SLURM not working properly")
            return False, "SLURM command failed"
    except FileNotFoundError:
        print("  ✗ SLURM not found")
        return False, "SLURM not installed"


def check_permissions():
    """Check file permissions and directories."""
    print("\nChecking permissions...")
    
    # Check if we can write to scratch directory
    scratch_dir = Path.home() / "scratch"
    if scratch_dir.exists():
        if os.access(scratch_dir, os.W_OK):
            print(f"  ✓ Can write to {scratch_dir}")
        else:
            print(f"  ✗ Cannot write to {scratch_dir}")
            return False, f"No write permission to {scratch_dir}"
    else:
        print(f"  ! Scratch directory {scratch_dir} doesn't exist (will be created)")
    
    # Check current directory permissions
    if os.access(".", os.W_OK):
        print("  ✓ Can write to current directory")
    else:
        print("  ✗ Cannot write to current directory")
        return False, "No write permission to current directory"
    
    return True, None


def main():
    print("="*60)
    print("MEMORY PROFILING EXPERIMENT SETUP CHECK")
    print("="*60)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    all_checks = [
        ("Required files", check_files),
        ("Python packages", check_python_packages), 
        ("Phage transformer package", check_phage_transformer_package),
        ("GPU availability", check_gpu),
        ("SLURM availability", check_slurm),
        ("Permissions", check_permissions)
    ]
    
    failed_checks = []
    
    for check_name, check_func in all_checks:
        try:
            success, error = check_func()
            if not success:
                failed_checks.append((check_name, error))
        except Exception as e:
            failed_checks.append((check_name, f"Check failed with error: {e}"))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not failed_checks:
        print("✓ All checks passed! You're ready to run the memory profiling experiment.")
        print("\nTo submit the job:")
        print("  python run_memory_experiment.py --submit")
        print("\nTo run locally (if you have GPU access):")
        print("  python run_memory_experiment.py --local")
    else:
        print("✗ Some checks failed. Please fix the following issues:")
        for check_name, error in failed_checks:
            print(f"\n{check_name}:")
            print(f"  Error: {error}")
        
        print("\n" + "-"*40)
        print("TROUBLESHOOTING TIPS")
        print("-"*40)
        
        if any("phage" in error.lower() for _, error in failed_checks):
            print("\nPhage transformer package issues:")
            print("  1. Make sure you're in the package root directory")
            print("  2. Or install the package: pip install -e .")
            print("  3. Or run from parent directory containing the package")
        
        if any("cuda" in error.lower() or "gpu" in error.lower() for _, error in failed_checks):
            print("\nGPU issues:")
            print("  1. Make sure you're on a GPU node")
            print("  2. Load CUDA modules if needed")
            print("  3. Check nvidia-smi output")
        
        if any("slurm" in error.lower() for _, error in failed_checks):
            print("\nSLURM issues:")
            print("  1. Make sure you're on a SLURM cluster")
            print("  2. Load required modules")
            print("  3. Check your account/partition settings in memory_profiling.slurm")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
