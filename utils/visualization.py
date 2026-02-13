"""
Visualization Module

Plotting utilities for training metrics and model comparison.
Generates publication-quality figures for learning curves and performance analysis.
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


# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------


def get_results_dir():
    """Return the base results directory path."""
    return Path("results")


def load_metrics(experiment_name):
    """
    Load training metrics from a completed experiment.

    Args:
        experiment_name: Name of the experiment directory

    Returns:
        pd.DataFrame: Metrics with columns [epoch, train_loss, val_loss,
                      val_acc_top1, val_acc_top3, lr]

    Raises:
        FileNotFoundError: If metrics.csv doesn't exist
    """
    metrics_path = get_results_dir() / experiment_name / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics found at {metrics_path}")
    return pd.read_csv(metrics_path)


def load_results(experiment_name):
    """
    Load final results from a completed experiment.

    Args:
        experiment_name: Name of the experiment directory

    Returns:
        dict: Results including test accuracy, inference time, etc.

    Raises:
        FileNotFoundError: If results.json doesn't exist
    """
    results_path = get_results_dir() / experiment_name / "results.json"
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


def plot_loss_curves(experiment_name, save=True, show=False):
    """
    Plot training and validation loss curves over epochs.

    Args:
        experiment_name: Name of the experiment
        save: Whether to save the figure to disk
        show: Whether to display the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    metrics = load_metrics(experiment_name)

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
            save_path = get_results_dir() / experiment_name / "loss_curves.png"
            fig.savefig(save_path)
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_accuracy_curves(experiment_name, save=True, show=False):
    """
    Plot top-1 and top-3 validation accuracy over epochs.

    Args:
        experiment_name: Name of the experiment
        save: Whether to save the figure to disk
        show: Whether to display the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    metrics = load_metrics(experiment_name)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(metrics['epoch'], metrics['val_acc_top1'] * 100,
                color=COLORS[0], linewidth=2, label='Top-1 Accuracy')
        ax.plot(metrics['epoch'], metrics['val_acc_top3'] * 100,
                color=COLORS[2], linewidth=2, label='Top-3 Accuracy')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Validation Accuracy - {experiment_name}')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)

        plt.tight_layout()

        if save:
            save_path = get_results_dir() / experiment_name / "accuracy_curves.png"
            fig.savefig(save_path)
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_learning_rate(experiment_name, save=True, show=False):
    """
    Plot learning rate schedule over epochs.

    Args:
        experiment_name: Name of the experiment
        save: Whether to save the figure to disk
        show: Whether to display the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    metrics = load_metrics(experiment_name)

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
            save_path = get_results_dir() / experiment_name / "learning_rate.png"
            fig.savefig(save_path)
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_experiment_summary(experiment_name, save=True, show=False):
    """
    Generate a 2x2 grid summarizing all training metrics.

    Includes: loss curves, accuracy curves, learning rate, and metrics table.

    Args:
        experiment_name: Name of the experiment
        save: Whether to save the figure to disk
        show: Whether to display the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    metrics = load_metrics(experiment_name)

    try:
        results = load_results(experiment_name)
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

        # Top-right: Accuracy curves
        ax = axes[0, 1]
        ax.plot(metrics['epoch'], metrics['val_acc_top1'] * 100,
                color=COLORS[0], linewidth=2, label='Top-1')
        ax.plot(metrics['epoch'], metrics['val_acc_top3'] * 100,
                color=COLORS[2], linewidth=2, label='Top-3')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Validation Accuracy')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)

        # Bottom-left: Learning rate
        ax = axes[1, 0]
        ax.plot(metrics['epoch'], metrics['lr'],
                color=COLORS[4], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

        # Bottom-right: Results table
        ax = axes[1, 1]
        ax.axis('off')

        if has_results:
            table_data = [
                ['Metric', 'Value'],
                ['Best Val Top-1', f"{results['best_val_acc_top1']:.2%}"],
                ['Best Val Top-3', f"{results['best_val_acc_top3']:.2%}"],
                ['Test Top-1', f"{results['final_test_acc_top1']:.2%}"],
                ['Test Top-3', f"{results['final_test_acc_top3']:.2%}"],
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
            save_path = get_results_dir() / experiment_name / "summary.png"
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

        x = range(len(valid_names))
        bar_width = 0.6

        # Test accuracy (Top-1 and Top-3)
        ax = axes[0]
        top1_vals = [r['final_test_acc_top1'] * 100 for r in results_list]
        top3_vals = [r['final_test_acc_top3'] * 100 for r in results_list]

        x_pos = range(len(valid_names))
        ax.bar([p - 0.15 for p in x_pos], top1_vals, width=0.3,
               color=COLORS[0], label='Top-1')
        ax.bar([p + 0.15 for p in x_pos], top3_vals, width=0.3,
               color=COLORS[2], label='Top-3')

        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Test Accuracy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_names, rotation=45, ha='right')
        ax.legend()
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
    """Command-line interface for visualization module."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize training metrics and compare model performance'
    )
    parser.add_argument(
        'experiment',
        nargs='?',
        help='Experiment name to visualize'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        metavar='EXP',
        help='Compare multiple experiments'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments'
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

    if args.compare:
        print(f"Comparing experiments: {', '.join(args.compare)}")
        plot_loss_comparison(args.compare, save=True, show=args.show)
        plot_accuracy_comparison(args.compare, save=True, show=args.show)
        plot_model_comparison_bar(args.compare, save=True, show=args.show)
        print("\nComparison plots saved to results/comparisons/")
        return

    if args.experiment:
        print(f"Generating plots for: {args.experiment}")
        plot_loss_curves(args.experiment, save=True, show=args.show)
        plot_accuracy_curves(args.experiment, save=True, show=args.show)
        plot_learning_rate(args.experiment, save=True, show=args.show)
        plot_experiment_summary(args.experiment, save=True, show=args.show)
        print(f"\nPlots saved to results/{args.experiment}/")
        return

    parser.print_help()


if __name__ == '__main__':
    main()
