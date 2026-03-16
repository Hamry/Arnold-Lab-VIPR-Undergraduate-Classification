#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization Script

Runs automated hyperparameter search using Optuna with SQLite storage
for study persistence and resumability.

Usage:
    python run_optuna.py configs/optuna/resnet152_sweep.json
    python run_optuna.py configs/optuna/resnet152_sweep.json --resume
"""

import argparse
import json
import shutil
import torch
from pathlib import Path
from datetime import datetime

import optuna
from optuna.storages import RDBStorage

from utils.trainer import train_model
from utils.optuna_utils import (
    generate_trial_name,
    sample_hyperparameters,
    apply_hyperparameters,
    create_pruner,
    cleanup_pruned_trial,
    cleanup_trials,
    TrialCleanupCallback,
    BestModelCallback,
    StudySummaryCallback,
)


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_optuna_config(config_path: str) -> dict:
    """
    Load and validate Optuna configuration.

    Args:
        config_path: Path to Optuna config JSON file

    Returns:
        Validated configuration dictionary

    Raises:
        ValueError: If required fields are missing
        FileNotFoundError: If config file doesn't exist
    """
    with open(config_path, "r") as f:
        optuna_config = json.load(f)

    # Validate required fields
    required = ["study_name", "base_config_path", "search_space"]
    for field in required:
        if field not in optuna_config:
            raise ValueError(f"Missing required field in Optuna config: {field}")

    # Validate base config exists
    if not Path(optuna_config["base_config_path"]).exists():
        raise FileNotFoundError(
            f"Base config not found: {optuna_config['base_config_path']}"
        )

    # Set defaults
    optuna_config.setdefault("n_trials", 50)
    optuna_config.setdefault("timeout_hours", None)
    optuna_config.setdefault("pruning", {"enabled": True})
    optuna_config.setdefault("cleanup", {"keep_top_n": 5, "cleanup_frequency": 10})

    return optuna_config


def load_base_config(base_config_path: str) -> dict:
    """
    Load the base training configuration.

    Args:
        base_config_path: Path to base config JSON file

    Returns:
        Base configuration dictionary
    """
    with open(base_config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Objective Function
# ---------------------------------------------------------------------------


def create_objective(optuna_config: dict, base_config: dict, study_dir: Path):
    """
    Create the Optuna objective function.

    The objective function:
    1. Samples hyperparameters from the search space
    2. Applies them to the base config
    3. Runs training with pruning support
    4. Returns validation accuracy as the objective value

    Args:
        optuna_config: Optuna configuration dictionary
        base_config: Base training configuration
        study_dir: Path to study results directory

    Returns:
        Callable objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        params = sample_hyperparameters(trial, optuna_config["search_space"])

        # Generate trial name
        trial_name = generate_trial_name(
            optuna_config["study_name"],
            trial.number,
            params,
        )
        trial.set_user_attr("trial_name", trial_name)

        print(f"\n{'=' * 60}")
        print(f"[Trial {trial.number}] {trial_name}")
        print(f"{'=' * 60}")
        print("Hyperparameters:")
        for name, value in params.items():
            print(f"  {name}: {value}")
        print()

        # Apply hyperparameters to config
        config = apply_hyperparameters(base_config, params)
        config["experiment_name"] = trial_name

        # Create trial directory
        trial_dir = study_dir / "trials" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Run training with pruning support
            results = train_model(
                config,
                trial=trial,
                results_dir_override=str(trial_dir),
            )

            print(f"\n[Trial {trial.number}] Completed with val_acc={results['best_val_acc_top1']:.4f}")

            # Return objective value (validation accuracy to maximize)
            return results["best_val_acc_top1"]

        except optuna.TrialPruned:
            print(f"\n[Trial {trial.number}] Pruned")
            cleanup_pruned_trial(trial_dir)
            raise

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[Trial {trial.number}] CUDA OOM - marking as pruned")
            torch.cuda.empty_cache()
            # Save error info
            error_path = trial_dir / "error.txt"
            with open(error_path, "w") as f:
                f.write(f"CUDA_OOM: {e}")
            raise optuna.TrialPruned()

        except Exception as e:
            print(f"\n[Trial {trial.number}] Error: {type(e).__name__}: {e}")
            # Save error info
            error_path = trial_dir / "error.txt"
            with open(error_path, "w") as f:
                f.write(f"{type(e).__name__}: {e}")
            # Mark as pruned to continue study
            raise optuna.TrialPruned()

    return objective


# ---------------------------------------------------------------------------
# Study Management
# ---------------------------------------------------------------------------


def create_or_load_study(
    optuna_config: dict, study_dir: Path, resume: bool = False
) -> optuna.Study:
    """
    Create a new study or load existing one.

    Uses SQLite storage for persistence and resumability.

    Args:
        optuna_config: Optuna configuration
        study_dir: Directory for study results
        resume: Whether to resume existing study

    Returns:
        optuna.Study object
    """
    study_name = optuna_config["study_name"]
    db_path = study_dir / "study.db"

    storage = RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )

    pruner = create_pruner(optuna_config)

    if resume and db_path.exists():
        print(f"[Optuna] Resuming study: {study_name}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
        print(f"[Optuna] Found {len(study.trials)} existing trials")
    else:
        print(f"[Optuna] Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",  # Maximize validation accuracy
            pruner=pruner,
            load_if_exists=resume,
        )

    return study


def run_optimization(optuna_config: dict, resume: bool = False):
    """
    Run the full Optuna optimization.

    Args:
        optuna_config: Optuna configuration dictionary
        resume: Whether to resume existing study
    """
    study_name = optuna_config["study_name"]

    # Setup directories
    study_dir = Path("results") / "optuna_studies" / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "trials").mkdir(exist_ok=True)

    # Save optuna config
    config_copy_path = study_dir / "optuna_config.json"
    with open(config_copy_path, "w") as f:
        json.dump(optuna_config, f, indent=2)

    # Load base config
    base_config = load_base_config(optuna_config["base_config_path"])

    # Create or load study
    study = create_or_load_study(optuna_config, study_dir, resume)

    # Create objective function
    objective = create_objective(optuna_config, base_config, study_dir)

    # Create callbacks
    cleanup_config = optuna_config.get("cleanup", {})
    callbacks = [
        TrialCleanupCallback(
            study_dir,
            keep_top_n=cleanup_config.get("keep_top_n", 5),
            frequency=cleanup_config.get("cleanup_frequency", 10),
        ),
        BestModelCallback(study_dir),
        StudySummaryCallback(study_dir, optuna_config),
    ]

    # Determine trial count and timeout
    n_trials = optuna_config.get("n_trials", 50)
    timeout_hours = optuna_config.get("timeout_hours")
    timeout_seconds = timeout_hours * 3600 if timeout_hours else None

    # Print study info
    print(f"\n{'=' * 60}")
    print(f"OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"{'=' * 60}")
    print(f"Study name: {study_name}")
    print(f"Base config: {optuna_config['base_config_path']}")
    print(f"Trials: {n_trials}")
    print(f"Timeout: {timeout_hours} hours" if timeout_hours else "Timeout: None")
    print(f"Search space: {len(optuna_config['search_space'])} parameters")
    print(f"Results dir: {study_dir}")
    print(f"{'=' * 60}\n")

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        callbacks=callbacks,
        gc_after_trial=True,  # Help with memory management
    )

    # Final cleanup to ensure only top-N trials remain after optimization ends
    print("[Optuna] Running final cleanup...")
    cleanup_trials(study_dir, study, keep_top_n=cleanup_config.get("keep_top_n", 5))

    # Print final summary
    print_study_summary(study, study_dir)


def print_study_summary(study: optuna.Study, study_dir: Path):
    """
    Print final study summary.

    Args:
        study: Completed Optuna study
        study_dir: Path to study results directory
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"\n{'=' * 60}")
    print("OPTUNA STUDY COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed: {len(completed)}")
    print(f"Pruned: {len(pruned)}")

    if study.best_trial:
        print(f"\nBest Trial: {study.best_trial.number}")
        print(f"Best Value: {study.best_trial.value:.4f}")
        print("Best Parameters:")
        for name, value in study.best_trial.params.items():
            print(f"  {name}: {value}")

        best_dir = study_dir / "best"
        if best_dir.exists():
            print(f"\nBest model saved to: {best_dir}")

    # Parameter importance (if enough trials)
    if len(completed) >= 10:
        try:
            print("\nParameter Importance:")
            importance = optuna.importance.get_param_importances(study)
            for name, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {name}: {imp:.3f}")
        except Exception:
            pass

    print(f"\nStudy summary: {study_dir / 'study_summary.json'}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start new study
    python run_optuna.py configs/optuna/resnet152_sweep.json

    # Resume existing study
    python run_optuna.py configs/optuna/resnet152_sweep.json --resume
        """,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to Optuna configuration JSON file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing study if available",
    )

    args = parser.parse_args()

    # Load config and run
    print(f"[Optuna] Loading config: {args.config}")
    optuna_config = load_optuna_config(args.config)
    run_optimization(optuna_config, resume=args.resume)


if __name__ == "__main__":
    main()
