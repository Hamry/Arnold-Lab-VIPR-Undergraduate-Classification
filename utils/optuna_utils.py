"""
Optuna Utility Functions

Helper functions for hyperparameter optimization with Optuna.
"""

import copy
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import optuna
from optuna.trial import TrialState


# ---------------------------------------------------------------------------
# Trial Naming
# ---------------------------------------------------------------------------


def generate_trial_hash(hyperparams: dict) -> str:
    """
    Generate 8-character hash from hyperparameters.

    Args:
        hyperparams: Dictionary of hyperparameter values

    Returns:
        8-character hex string (first 8 chars of MD5)
    """
    sorted_params = json.dumps(hyperparams, sort_keys=True, default=str)
    return hashlib.md5(sorted_params.encode()).hexdigest()[:8]


def generate_trial_name(study_name: str, trial_number: int, hyperparams: dict) -> str:
    """
    Generate trial name: {study}_trial{num}_{hash}.

    Args:
        study_name: Name of the Optuna study
        trial_number: Trial number (0-indexed from Optuna)
        hyperparams: Dictionary of sampled hyperparameters

    Returns:
        Unique trial name string
    """
    param_hash = generate_trial_hash(hyperparams)
    return f"{study_name}_trial{trial_number:03d}_{param_hash}"


# ---------------------------------------------------------------------------
# Hyperparameter Sampling
# ---------------------------------------------------------------------------


def sample_hyperparameters(trial: optuna.Trial, search_space: dict) -> dict:
    """
    Sample hyperparameters from search space using Optuna trial.

    Supports parameter types:
    - float: Uniform float distribution (low, high)
    - log_float: Log-uniform float distribution (low, high)
    - int: Uniform integer distribution (low, high, optional step)
    - log_int: Log-uniform integer distribution (low, high)
    - categorical: Categorical choices (choices list)

    Args:
        trial: Optuna trial object
        search_space: Dictionary defining search space

    Returns:
        Dictionary of sampled hyperparameter values
    """
    params = {}

    for param_name, param_config in search_space.items():
        param_type = param_config["type"]

        if param_type == "float":
            params[param_name] = trial.suggest_float(
                param_name, param_config["low"], param_config["high"]
            )
        elif param_type == "log_float":
            params[param_name] = trial.suggest_float(
                param_name, param_config["low"], param_config["high"], log=True
            )
        elif param_type == "int":
            step = param_config.get("step", 1)
            params[param_name] = trial.suggest_int(
                param_name, param_config["low"], param_config["high"], step=step
            )
        elif param_type == "log_int":
            params[param_name] = trial.suggest_int(
                param_name, param_config["low"], param_config["high"], log=True
            )
        elif param_type == "categorical":
            choices = param_config["choices"]
            # Handle complex types (lists, dicts) by using index selection
            if any(isinstance(c, (list, dict)) for c in choices):
                choice_idx = trial.suggest_categorical(
                    f"{param_name}_idx", list(range(len(choices)))
                )
                params[param_name] = choices[choice_idx]
            else:
                params[param_name] = trial.suggest_categorical(param_name, choices)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return params


def apply_hyperparameters(base_config: dict, params: dict) -> dict:
    """
    Apply sampled hyperparameters to base config.

    Supports nested keys using dot notation (e.g., "training.learning_rate").

    Special handling:
    - training.thaw_epoch + training.thaw_percent: Converted to thaw_schedule dict
      Example: thaw_epoch=15, thaw_percent=0.5 -> thaw_schedule={"15": 0.5}

    Args:
        base_config: Base configuration dictionary
        params: Sampled hyperparameters

    Returns:
        Modified configuration dictionary (deep copy)
    """
    config = copy.deepcopy(base_config)

    for param_name, value in params.items():
        keys = param_name.split(".")
        target = config

        # Navigate to nested key
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]

        # Set value
        target[keys[-1]] = value

    # Special handling: convert thaw_epoch + thaw_percent to thaw_schedule
    training = config.get("training", {})
    if "thaw_epoch" in training and "thaw_percent" in training:
        thaw_epoch = training.pop("thaw_epoch")
        thaw_percent = training.pop("thaw_percent")
        training["thaw_schedule"] = {str(thaw_epoch): thaw_percent}

    return config


# ---------------------------------------------------------------------------
# Pruner Creation
# ---------------------------------------------------------------------------


def create_pruner(optuna_config: dict) -> optuna.pruners.BasePruner:
    """
    Create Optuna pruner from config.

    Uses PercentilePruner (default bottom 25%) so that only the worst-performing
    trials are pruned.  For dynamic-unfreeze runs, pruning checkpoints fire right
    before each backbone block is thawed (step = unfreeze event index), making
    cross-trial comparisons at each step more meaningful than epoch-by-epoch
    comparisons.  For non-dynamic runs, pruning checkpoints fire every epoch as
    before.

    Config keys (all optional):
        percentile        – prune if in bottom N% (default 25.0)
        n_startup_trials  – trials to run before pruning starts (default 5)
        n_warmup_steps    – reported steps to skip before pruning each trial (default 0)
        interval_steps    – check every N steps (default 1)

    Args:
        optuna_config: Optuna configuration dictionary

    Returns:
        Optuna pruner object
    """
    pruning = optuna_config.get("pruning", {})

    if not pruning.get("enabled", True):
        return optuna.pruners.NopPruner()

    return optuna.pruners.PercentilePruner(
        percentile=pruning.get("percentile", 25.0),
        n_startup_trials=pruning.get("n_startup_trials", 5),
        n_warmup_steps=pruning.get("n_warmup_steps", 0),
        interval_steps=pruning.get("interval_steps", 1),
    )


# ---------------------------------------------------------------------------
# Study Summary
# ---------------------------------------------------------------------------


def update_study_summary(study_dir: Path, study: optuna.Study, config: dict):
    """
    Update study summary JSON file.

    Saves study metadata, best trial info, top trials, and parameter importance.

    Args:
        study_dir: Path to study results directory
        study: Optuna study object
        config: Optuna configuration dictionary
    """
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == TrialState.FAIL]

    summary = {
        "study_name": config["study_name"],
        "updated_at": datetime.now().isoformat(),
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
        "n_trials_failed": len(failed),
        "n_trials_total": len(study.trials),
    }

    if study.best_trial:
        summary["best_trial"] = {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "trial_name": study.best_trial.user_attrs.get("trial_name"),
        }

    # Top trials
    top_5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    summary["top_5_trials"] = [
        {
            "number": t.number,
            "value": t.value,
            "trial_name": t.user_attrs.get("trial_name"),
        }
        for t in top_5
    ]

    # Parameter importance (if enough trials)
    if len(completed) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            summary["parameter_importance"] = importance
        except Exception:
            pass

    with open(study_dir / "study_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Trial Cleanup
# ---------------------------------------------------------------------------


def cleanup_trials(study_dir: Path, study: optuna.Study, keep_top_n: int = 5):
    """
    Remove trial directories for non-top trials.

    Preserves:
    - Top N trials by objective value
    - Best trial (always)

    Args:
        study_dir: Path to study results directory
        study: Optuna study object
        keep_top_n: Number of top trials to preserve
    """
    trials_dir = study_dir / "trials"
    top_n_dir = study_dir / f"top_{keep_top_n}"

    if not trials_dir.exists():
        return

    # Get completed trials sorted by value (descending for maximize)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)
    top_numbers = {t.number for t in sorted_trials[:keep_top_n]}

    # Recreate top_n directory
    if top_n_dir.exists():
        shutil.rmtree(top_n_dir)
    top_n_dir.mkdir(parents=True)

    # Process trials
    for trial in completed:
        trial_name = trial.user_attrs.get("trial_name")
        if not trial_name:
            continue

        trial_dir = trials_dir / trial_name

        if trial.number in top_numbers:
            # Copy to top_n directory
            if trial_dir.exists():
                dest = top_n_dir / trial_name
                if not dest.exists():
                    shutil.copytree(trial_dir, dest)
        else:
            # Remove non-top trial directory
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
                print(f"[Cleanup] Removed: {trial_name}")


def cleanup_pruned_trial(trial_dir: Path):
    """
    Clean up a pruned trial's directory.

    Args:
        trial_dir: Path to trial directory
    """
    if trial_dir.exists():
        shutil.rmtree(trial_dir)


# ---------------------------------------------------------------------------
# Best Model Management
# ---------------------------------------------------------------------------


def update_best_model(study_dir: Path, trial: optuna.trial.FrozenTrial, study: optuna.Study):
    """
    Update best model directory if this trial is the new best.

    Args:
        study_dir: Path to study results directory
        trial: Completed trial
        study: Optuna study object
    """
    if study.best_trial is None:
        return

    if study.best_trial.number != trial.number:
        return

    best_dir = study_dir / "best"
    trial_name = trial.user_attrs.get("trial_name")

    if not trial_name:
        return

    trial_dir = study_dir / "trials" / trial_name

    if not trial_dir.exists():
        return

    # Clear existing best directory
    if best_dir.exists():
        shutil.rmtree(best_dir)

    # Copy trial results to best directory
    shutil.copytree(trial_dir, best_dir)
    print(f"[Optuna] New best trial: {trial.number} with value {trial.value:.4f}")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TrialCleanupCallback:
    """Optuna callback to periodically clean up non-performant trials."""

    def __init__(self, study_dir: Path, keep_top_n: int = 5, frequency: int = 10):
        """
        Initialize cleanup callback.

        Args:
            study_dir: Path to study results directory
            keep_top_n: Number of top trials to preserve
            frequency: Cleanup after every N completed trials
        """
        self.study_dir = study_dir
        self.keep_top_n = keep_top_n
        self.frequency = frequency
        self._completed_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == TrialState.COMPLETE:
            self._completed_count += 1

            if self._completed_count % self.frequency == 0:
                print(f"[Cleanup] Running cleanup after {self._completed_count} completed trials...")
                cleanup_trials(self.study_dir, study, self.keep_top_n)


class BestModelCallback:
    """Optuna callback to update best model after each trial."""

    def __init__(self, study_dir: Path):
        """
        Initialize best model callback.

        Args:
            study_dir: Path to study results directory
        """
        self.study_dir = study_dir

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == TrialState.COMPLETE:
            update_best_model(self.study_dir, trial, study)


class StudySummaryCallback:
    """Optuna callback to update study summary after each trial."""

    def __init__(self, study_dir: Path, config: dict):
        """
        Initialize summary callback.

        Args:
            study_dir: Path to study results directory
            config: Optuna configuration dictionary
        """
        self.study_dir = study_dir
        self.config = config

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        update_study_summary(self.study_dir, study, self.config)
