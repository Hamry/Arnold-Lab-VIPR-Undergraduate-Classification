"""
Batch Experiment Runner

Runs multiple experiments sequentially from a list of config files.

Usage:
    python run_batch.py configs/resnet152.json configs/vgg16.json configs/inception_v3.json
    python run_batch.py configs/*.json
"""

import sys
import json
import time
from pathlib import Path
from utils import train_model
import torch


def run_experiment(config_path):
    """Run a single experiment from a config file."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {config_path}")
    print(f"{'='*60}\n")

    with open(config_path, "r") as f:
        options = json.load(f)

    start_time = time.time()
    results = train_model(options)
    elapsed = time.time() - start_time

    print(f"\nExperiment completed in {elapsed/60:.1f} minutes")
    print(f"Results saved to: results/{options['experiment_name']}/")
    torch.cuda.empty_cache()
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_batch.py <config1.json> [config2.json] ...")
        print("Example: python run_batch.py configs/resnet152.json configs/vgg16.json")
        sys.exit(1)

    config_paths = sys.argv[1:]

    # Validate all config files exist before starting
    for path in config_paths:
        if not Path(path).exists():
            print(f"Error: Config file not found: {path}")
            sys.exit(1)

    print(f"Running {len(config_paths)} experiments:")
    for i, path in enumerate(config_paths, 1):
        print(f"  {i}. {path}")

    all_results = []
    for i, config_path in enumerate(config_paths, 1):
        print(f"\n[{i}/{len(config_paths)}] ", end="")
        try:
            results = run_experiment(config_path)
            all_results.append(
                {"config": config_path, "results": results, "status": "success"}
            )
        except Exception as e:
            print(f"Error running {config_path}: {e}")
            all_results.append(
                {"config": config_path, "error": str(e), "status": "failed"}
            )

    # Summary
    print(f"\n{'='*60}")
    print("Batch Summary")
    print(f"{'='*60}")
    successful = sum(1 for r in all_results if r["status"] == "success")
    print(f"Completed: {successful}/{len(config_paths)} experiments")

    for r in all_results:
        status = "OK" if r["status"] == "success" else "FAILED"
        print(f"  [{status}] {r['config']}")


if __name__ == "__main__":
    main()
