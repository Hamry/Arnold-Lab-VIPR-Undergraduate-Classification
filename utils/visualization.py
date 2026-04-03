"""
Visualization Module

Plotting utilities for training metrics and model comparison.
Single-experiment plots are auto-generated during training.
This CLI focuses on comparing multiple experiments.

Usage:
    python -m utils.visualization exp1 exp2    # Compare experiments
    python -m utils.visualization --all        # Compare all experiments
    python -m utils.visualization --list       # List available experiments
    python -m utils.visualization --regenerate exp1  # Regenerate single plots
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Colorblind-friendly palette (IBM Design)
COLORS = [
    '#648FFF',  # Blue
    '#DC267F',  # Magenta
    '#FFB000',  # Gold
    '#FE6100',  # Orange
    '#785EF0',  # Purple
    '#009E73',  # Teal
]

# Publication-quality plot settings
PLOT_STYLE = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# IEEE-style publication settings (Times New Roman, inward ticks, minimal grid)
IEEE_PLOT_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'legend.fontsize': 8,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
}


# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------


def get_results_dir():
    """Return the base results directory path."""
    return Path("results")


def load_metrics(experiment_name, results_dir=None):
    """
    Load training metrics from a completed experiment.

    Args:
        experiment_name: Name of the experiment directory
        results_dir: Optional explicit path to results directory

    Returns:
        pd.DataFrame: Metrics with columns [epoch, train_loss, val_loss,
                      val_acc_top1, lr]

    Raises:
        FileNotFoundError: If metrics.csv doesn't exist
    """
    if results_dir is None:
        results_dir = get_results_dir() / experiment_name
    else:
        results_dir = Path(results_dir)
    metrics_path = results_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics found at {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    # Backward compatibility: drop top-3 column if present
    if 'val_acc_top3' in metrics.columns:
        metrics = metrics.drop(columns=['val_acc_top3'])
    return metrics


def load_results(experiment_name, results_dir=None):
    """
    Load final results from a completed experiment.

    Args:
        experiment_name: Name of the experiment directory
        results_dir: Optional explicit path to results directory

    Returns:
        dict: Results including test accuracy, inference time, etc.

    Raises:
        FileNotFoundError: If results.json doesn't exist
    """
    if results_dir is None:
        results_dir = get_results_dir() / experiment_name
    else:
        results_dir = Path(results_dir)
    results_path = results_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results found at {results_path}")
    with open(results_path) as f:
        return json.load(f)


def discover_experiments():
    """
    Find all experiments with valid metrics files.

    Returns:
        list[str]: Sorted list of experiment names
    """
    results_dir = get_results_dir()
    if not results_dir.exists():
        return []

    experiments = []
    for path in results_dir.iterdir():
        if path.is_dir() and (path / "metrics.csv").exists():
            experiments.append(path.name)

    return sorted(experiments)


# ---------------------------------------------------------------------------
# Single Experiment Plots
# ---------------------------------------------------------------------------


def plot_loss_curves(experiment_name, save=True, show=False, results_dir=None):
    """
    Plot training and validation loss curves over epochs.

    Args:
        experiment_name: Name of the experiment
        save: Whether to save the figure to disk
        show: Whether to display the figure
        results_dir: Optional explicit path to results directory

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if results_dir is None:
        results_dir = get_results_dir() / experiment_name
    else:
        results_dir = Path(results_dir)

    metrics = load_metrics(experiment_name, results_dir=results_dir)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(metrics['epoch'], metrics['train_loss'],
                color=COLORS[0], linewidth=2, label='Train Loss')
        ax.plot(metrics['epoch'], metrics['val_loss'],
                color=COLORS[1], linewidth=2, label='Validation Loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss Curves - {experiment_name}')
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save:
            save_path = results_dir / "loss_curves.png"
            fig.savefig(save_path)
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_accuracy_curves(experiment_name, save=True, show=False, results_dir=None):
    """
    Plot validation accuracy over epochs.

    Args:
        experiment_name: Name of the experiment
        save: Whether to save the figure to disk
        show: Whether to display the figure
        results_dir: Optional explicit path to results directory

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if results_dir is None:
        results_dir = get_results_dir() / experiment_name
    else:
        results_dir = Path(results_dir)

    metrics = load_metrics(experiment_name, results_dir=results_dir)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(metrics['epoch'], metrics['val_acc_top1'] * 100,
                color=COLORS[0], linewidth=2, label='Validation Accuracy')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Validation Accuracy - {experiment_name}')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)

        plt.tight_layout()

        if save:
            save_path = results_dir / "accuracy_curves.png"
            fig.savefig(save_path)
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_learning_rate(experiment_name, save=True, show=False, results_dir=None):
    """
    Plot learning rate schedule over epochs.

    Args:
        experiment_name: Name of the experiment
        save: Whether to save the figure to disk
        show: Whether to display the figure
        results_dir: Optional explicit path to results directory

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if results_dir is None:
        results_dir = get_results_dir() / experiment_name
    else:
        results_dir = Path(results_dir)

    metrics = load_metrics(experiment_name, results_dir=results_dir)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(metrics['epoch'], metrics['lr'],
                color=COLORS[4], linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'Learning Rate Schedule - {experiment_name}')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

        plt.tight_layout()

        if save:
            save_path = results_dir / "learning_rate.png"
            fig.savefig(save_path)
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_confusion_matrix(all_targets, all_predictions, class_names,
                          results_dir, save=True, show=False):
    """
    Generate and save a confusion matrix plot.

    Args:
        all_targets: List/array of true labels
        all_predictions: List/array of predicted labels
        class_names: List of class names for axis labels
        results_dir: Path to save the figure
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    import numpy as np

    # Compute confusion matrix
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_targets, all_predictions):
        cm[t, p] += 1

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        # Set labels
        ax.set(xticks=np.arange(num_classes),
               yticks=np.arange(num_classes),
               xticklabels=class_names,
               yticklabels=class_names,
               xlabel='Predicted Label',
               ylabel='True Label',
               title='Confusion Matrix (Test Set)')

        # Rotate x labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                 rotation_mode='anchor')

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()

        if save:
            save_path = Path(results_dir) / "confusion_matrix.png"
            fig.savefig(save_path)
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_experiment_summary(experiment_name, save=True, show=False, results_dir=None):
    """
    Generate a 2x2 grid summarizing all training metrics.

    Includes: loss curves, accuracy curves, learning rate, and metrics table.

    Args:
        experiment_name: Name of the experiment
        save: Whether to save the figure to disk
        show: Whether to display the figure
        results_dir: Optional explicit path to results directory

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if results_dir is None:
        results_dir = get_results_dir() / experiment_name
    else:
        results_dir = Path(results_dir)

    metrics = load_metrics(experiment_name, results_dir=results_dir)

    try:
        results = load_results(experiment_name, results_dir=results_dir)
        has_results = True
    except FileNotFoundError:
        has_results = False

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Top-left: Loss curves
        ax = axes[0, 0]
        ax.plot(metrics['epoch'], metrics['train_loss'],
                color=COLORS[0], linewidth=2, label='Train Loss')
        ax.plot(metrics['epoch'], metrics['val_loss'],
                color=COLORS[1], linewidth=2, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend(loc='upper right')

        # Top-right: F1 score curves (train and val)
        ax = axes[0, 1]
        f1_cols = [c for c in metrics.columns if c.startswith('val_f1_')]
        if f1_cols:
            val_macro_f1 = metrics[f1_cols].mean(axis=1)
            ax.plot(metrics['epoch'], val_macro_f1,
                    color=COLORS[1], linewidth=2, label='Val F1')
        if 'train_macro_f1' in metrics.columns:
            ax.plot(metrics['epoch'], metrics['train_macro_f1'],
                    color=COLORS[0], linewidth=2, label='Train F1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)

        # Bottom-left: Per-class F1 score over epochs
        ax = axes[1, 0]
        f1_cols = [c for c in metrics.columns if c.startswith('val_f1_')]
        for i, col in enumerate(f1_cols):
            class_name = col.replace('val_f1_', '').capitalize()
            ax.plot(metrics['epoch'], metrics[col],
                    color=COLORS[i % len(COLORS)], linewidth=2, label=class_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)

        # Bottom-right: Results table
        ax = axes[1, 1]
        ax.axis('off')

        if has_results:
            f1_cols = [c for c in metrics.columns if c.startswith('val_f1_')]
            best_val_f1 = metrics[f1_cols].mean(axis=1).max() if f1_cols else None
            table_data = [
                ['Metric', 'Value'],
                ['Best Val F1', f"{best_val_f1:.4f}" if best_val_f1 is not None else 'N/A'],
                ['Test Acc', f"{results['final_test_acc_top1']:.2%}"],
                ['Best Epoch', str(results['best_epoch'])],
                ['Inference (ms)', f"{results['inference_time_ms']:.2f}"],
                ['Total Params', f"{results['total_params']:,}"],
                ['Trainable Params', f"{results['trainable_params']:,}"],
            ]

            table = ax.table(
                cellText=table_data[1:],
                colLabels=table_data[0],
                cellLoc='center',
                loc='center',
                colWidths=[0.5, 0.5]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            # Style header
            for i in range(2):
                table[(0, i)].set_facecolor('#E6E6E6')
                table[(0, i)].set_text_props(weight='bold')
        else:
            ax.text(0.5, 0.5, 'Results not available',
                    ha='center', va='center', fontsize=12)

        ax.set_title('Final Results')

        fig.suptitle(f'Experiment Summary: {experiment_name}',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            save_path = results_dir / "summary.png"
            fig.savefig(save_path)
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Multi-Model Comparison Plots
# ---------------------------------------------------------------------------


def _get_comparison_dir():
    """Get or create the comparisons output directory."""
    comparison_dir = get_results_dir() / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    return comparison_dir


def plot_loss_comparison(experiment_names, save=True, show=False):
    """
    Overlay validation loss curves from multiple experiments.

    Args:
        experiment_names: List of experiment names to compare
        save: Whether to save the figure to disk
        show: Whether to display the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, name in enumerate(experiment_names):
            metrics = load_metrics(name)
            color = COLORS[i % len(COLORS)]
            ax.plot(metrics['epoch'], metrics['val_loss'],
                    color=color, linewidth=2, label=name)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Comparison')
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save:
            save_path = _get_comparison_dir() / "loss_comparison.png"
            fig.savefig(save_path)
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_accuracy_comparison(experiment_names, save=True, show=False):
    """
    Overlay validation accuracy curves from multiple experiments.

    Args:
        experiment_names: List of experiment names to compare
        save: Whether to save the figure to disk
        show: Whether to display the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, name in enumerate(experiment_names):
            metrics = load_metrics(name)
            color = COLORS[i % len(COLORS)]
            ax.plot(metrics['epoch'], metrics['val_acc_top1'] * 100,
                    color=color, linewidth=2, label=name)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.set_title('Validation Accuracy Comparison')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)

        plt.tight_layout()

        if save:
            save_path = _get_comparison_dir() / "accuracy_comparison.png"
            fig.savefig(save_path)
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_model_comparison_bar(experiment_names, save=True, show=False):
    """
    Create bar chart comparing final test metrics across experiments.

    Args:
        experiment_names: List of experiment names to compare
        save: Whether to save the figure to disk
        show: Whether to display the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Load results for all experiments
    results_list = []
    valid_names = []
    for name in experiment_names:
        try:
            results = load_results(name)
            results_list.append(results)
            valid_names.append(name)
        except FileNotFoundError:
            print(f"Warning: No results.json for {name}, skipping")

    if not results_list:
        raise ValueError("No valid experiments with results.json found")

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        x_pos = range(len(valid_names))
        bar_width = 0.6

        # Test accuracy
        ax = axes[0]
        acc_vals = [r['final_test_acc_top1'] * 100 for r in results_list]

        ax.bar(x_pos, acc_vals, width=bar_width, color=COLORS[0])

        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Test Accuracy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_names, rotation=45, ha='right')
        ax.set_ylim(0, 105)

        # Inference time
        ax = axes[1]
        times = [r['inference_time_ms'] for r in results_list]
        ax.bar(x_pos, times, width=bar_width, color=COLORS[3])
        ax.set_xlabel('Model')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Inference Time')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_names, rotation=45, ha='right')

        # Trainable parameters
        ax = axes[2]
        params = [r['trainable_params'] / 1e6 for r in results_list]
        ax.bar(x_pos, params, width=bar_width, color=COLORS[4])
        ax.set_xlabel('Model')
        ax.set_ylabel('Parameters (M)')
        ax.set_title('Trainable Parameters')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_names, rotation=45, ha='right')

        fig.suptitle('Model Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save:
            save_path = _get_comparison_dir() / "model_comparison_bar.png"
            fig.savefig(save_path, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------


def main():
    """Command-line interface for comparing experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare training results across multiple experiments. '
                    'Single-experiment plots are auto-generated during training.'
    )
    parser.add_argument(
        'experiments',
        nargs='*',
        metavar='EXP',
        help='Experiment names to compare (requires 2+)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Compare all available experiments'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments'
    )
    parser.add_argument(
        '--regenerate',
        metavar='EXP',
        help='Regenerate single-experiment plots for a specific experiment'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively (in addition to saving)'
    )

    args = parser.parse_args()

    if args.list:
        experiments = discover_experiments()
        if experiments:
            print("Available experiments:")
            for exp in experiments:
                print(f"  - {exp}")
        else:
            print("No experiments found in results/")
        return

    # Regenerate single-experiment plots
    if args.regenerate:
        print(f"Regenerating plots for: {args.regenerate}")
        plot_loss_curves(args.regenerate, save=True, show=args.show)
        plot_accuracy_curves(args.regenerate, save=True, show=args.show)
        plot_learning_rate(args.regenerate, save=True, show=args.show)
        plot_experiment_summary(args.regenerate, save=True, show=args.show)
        print(f"\nPlots saved to results/{args.regenerate}/")
        return

    # Determine experiments to compare
    if args.all:
        experiments = discover_experiments()
        if not experiments:
            print("No experiments found in results/")
            return
    elif args.experiments:
        experiments = args.experiments
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m utils.visualization exp1 exp2     # Compare experiments")
        print("  python -m utils.visualization --all         # Compare all experiments")
        print("  python -m utils.visualization --regenerate exp1  # Regenerate plots")
        return

    if len(experiments) < 2:
        print("Error: Need at least 2 experiments to compare")
        print("Use --regenerate for single-experiment plots")
        return

    print(f"Comparing experiments: {', '.join(experiments)}")
    plot_loss_comparison(experiments, save=True, show=args.show)
    plot_accuracy_comparison(experiments, save=True, show=args.show)
    plot_model_comparison_bar(experiments, save=True, show=args.show)
    print("\nComparison plots saved to results/comparisons/")


if __name__ == '__main__':
    main()
