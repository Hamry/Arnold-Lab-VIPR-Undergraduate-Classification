"""
VIPR Classification Framework

Utilities for training and evaluating image classification models.
"""

from .model_utils import (
    load_model,
    count_parameters,
    BACKBONE_REGISTRY,
    get_resolution_probe,
    get_unfreeze_units,
    thaw_units,
    UnfreezeUnit,
)
from .trainer import train_model, DynamicThawController
from .visualization import (
    load_metrics,
    load_results,
    discover_experiments,
    plot_loss_curves,
    plot_accuracy_curves,
    plot_learning_rate,
    plot_experiment_summary,
    plot_loss_comparison,
    plot_accuracy_comparison,
    plot_model_comparison_bar,
    IEEE_PLOT_STYLE,
)

# Optuna utilities (optional, only available when optuna is installed)
try:
    from .optuna_utils import (
        generate_trial_hash,
        generate_trial_name,
        sample_hyperparameters,
        apply_hyperparameters,
        create_pruner,
        update_study_summary,
        cleanup_trials,
        cleanup_pruned_trial,
        update_best_model,
        TrialCleanupCallback,
        BestModelCallback,
        StudySummaryCallback,
    )
except ImportError:
    pass
