"""
Run Experiment Script

Simple entry point for running a single experiment from a config file.

Usage:
    python run_experiment.py configs/resnet152_baseline.json
"""

import sys
import json
from utils import train_model


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]

    with open(config_path, 'r') as f:
        options = json.load(f)

    results = train_model(options)

    print(f"\nResults saved to: results/{options['experiment_name']}/")


if __name__ == '__main__':
    main()
