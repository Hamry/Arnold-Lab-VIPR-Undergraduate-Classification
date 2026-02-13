"""
VIPR Classification Framework

Utilities for training and evaluating image classification models.
"""

from .model_utils import load_model, count_parameters, BACKBONE_REGISTRY
from .trainer import train_model
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
)
