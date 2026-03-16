"""
generate_paper_charts.py

Generates publication-quality figures from Optuna study results for use in
research papers. Reads from results/optuna_studies/{study_name}/best/ and
outputs numbered PNGs (and a CSV table) to results/paper_figures/.

Usage:
    python generate_paper_charts.py STUDY [STUDY ...]
        [--labels LABEL ...]         # display name overrides (default: backbone from config.json)
        [--output results/paper_figures]
        [--study-dir results/optuna_studies]

Example:
    python generate_paper_charts.py resnet152_fine_tuning_sweep efficientnet_sweep \\
        --labels "ResNet-152" "EfficientNet-B4"
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.visualization import COLORS, IEEE_PLOT_STYLE as PLOT_STYLE


# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------


def load_study_best(study_name, study_dir):
    """
    Load results.json, metrics.csv, and config.json for a study's best trial.

    Args:
        study_name: Optuna study directory name
        study_dir:  Path to the optuna_studies base directory

    Returns:
        dict with keys: study_name, best_dir, results, metrics, config, backbone
        Never raises — missing files produce warnings and empty/None values.
    """
    best_dir = Path(study_dir) / study_name / "best"
    out = {
        "study_name": study_name,
        "best_dir": best_dir,
        "results": {},
        "metrics": None,
        "config": {},
        "backbone": study_name,
    }

    results_path = best_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            out["results"] = json.load(f)
    else:
        print(f"[Warning] Missing results.json for study '{study_name}' at {results_path}")

    metrics_path = best_dir / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        if "val_acc_top3" in df.columns:
            df = df.drop(columns=["val_acc_top3"])
        out["metrics"] = df
    else:
        print(f"[Warning] Missing metrics.csv for study '{study_name}' at {metrics_path}")

    config_path = best_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            out["config"] = json.load(f)
        out["backbone"] = out["config"].get("model", {}).get("backbone", study_name)
    else:
        print(f"[Warning] Missing config.json for study '{study_name}' at {config_path}")

    return out



# ---------------------------------------------------------------------------
# Chart 1 — Overall Accuracy Bar Chart
# ---------------------------------------------------------------------------


def chart_01_accuracy_bar(study_data, labels, output_dir):
    """
    Bar chart of final test accuracy per architecture.

    Saves: output_dir/01_accuracy_by_architecture.png
    """
    x_labels, y_vals = [], []
    for datum, label in zip(study_data, labels):
        acc = datum["results"].get("final_test_acc_top1")
        if acc is None:
            print(f"[Chart 01] Skipping '{label}' — missing final_test_acc_top1")
            continue
        x_labels.append(label)
        y_vals.append(acc * 100)

    if not x_labels:
        print("[Chart 01] No data available — skipping")
        return

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 1.4), 5))

        x_pos = np.arange(len(x_labels))
        bars = ax.bar(x_pos, y_vals, width=0.6, color=COLORS[0])

        # Value labels above bars
        for bar, val in zip(bars, y_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 3 else 0, ha="right")
        ax.set_ylabel("Final Test Accuracy (%)")
        ax.set_title("Test Accuracy by Architecture")
        floor = max(0, min(y_vals) - 5)
        ax.set_ylim(floor, 101)

        plt.tight_layout()
        save_path = output_dir / "01_accuracy_by_architecture.png"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Chart 01] Saved: {save_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2 — Bubble Chart: Model Size vs Accuracy
# ---------------------------------------------------------------------------


def chart_02_bubble_size_accuracy(study_data, labels, output_dir):
    """
    Bubble chart: trainable params (M) vs test accuracy, bubble = inference time.

    Saves: output_dir/02_size_vs_accuracy.png
    """
    entries = []
    for datum, label in zip(study_data, labels):
        r = datum["results"]
        params = r.get("trainable_params")
        acc = r.get("final_test_acc_top1")
        time_ms = r.get("inference_time_ms")
        if None in (params, acc, time_ms):
            print(f"[Chart 02] Skipping '{label}' — missing size/acc/inference fields")
            continue
        entries.append({"label": label, "params_m": params / 1e6,
                        "acc": acc * 100, "time_ms": time_ms})

    if not entries:
        print("[Chart 02] No data available — skipping")
        return

    times = np.array([e["time_ms"] for e in entries])
    # Log-scale bubble sizes so large inference-time differences don't dominate;
    # maps log(1+t) linearly into [100, 1600] marker area units.
    log_times = np.log1p(times)
    max_log_t = log_times.max() if log_times.max() > 0 else 1.0
    sizes = (log_times / max_log_t) * 1500 + 100

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter_handles = []
        for i, (entry, sz) in enumerate(zip(entries, sizes)):
            sc = ax.scatter(
                entry["params_m"], entry["acc"],
                s=sz, color=COLORS[i % len(COLORS)],
                alpha=0.8, edgecolors="white", linewidths=0.5,
                label=entry["label"],
                zorder=3,
            )
            scatter_handles.append(sc)
            ax.annotate(
                entry["label"],
                (entry["params_m"], entry["acc"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

        # Log scale on x-axis so closely-spaced parameter counts are separable
        ax.set_xscale("log")

        # Model color legend — outside the plot to the right
        model_legend = ax.legend(
            handles=scatter_handles,
            loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0,
            title="Model", fontsize=8, title_fontsize=8,
        )

        # Bubble size legend (raw inference time values; bubble area is log-scaled)
        if len(entries) >= 2:
            t_min, t_mid, t_max = times.min(), np.median(times), times.max()
            lt_min   = np.log1p(t_min)
            lt_mid   = np.log1p(t_mid)
            lt_max   = np.log1p(t_max)
            s_min    = (lt_min   / max_log_t) * 1500 + 100
            s_mid    = (lt_mid   / max_log_t) * 1500 + 100
            s_max_sz = (lt_max   / max_log_t) * 1500 + 100
            size_handles = [
                ax.scatter([], [], s=s_min,    color="gray", alpha=0.6,
                           label=f"{t_min:.2f} ms"),
                ax.scatter([], [], s=s_mid,    color="gray", alpha=0.6,
                           label=f"{t_mid:.2f} ms"),
                ax.scatter([], [], s=s_max_sz, color="gray", alpha=0.6,
                           label=f"{t_max:.2f} ms"),
            ]
            ax.add_artist(model_legend)
            fig.legend(
                handles=size_handles,
                loc="lower left", bbox_to_anchor=(1.02, 0), borderaxespad=0,
                title="Inference Time", fontsize=8, title_fontsize=8,
            )

        ax.set_xlabel("Trainable Parameters (M, log scale)")
        ax.set_ylabel("Final Test Accuracy (%)")
        ax.set_title("Model Size vs. Accuracy\n(bubble size ∝ log inference time)")

        plt.tight_layout()
        save_path = output_dir / "02_size_vs_accuracy.png"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Chart 02] Saved: {save_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3 — Compound Learning Curves (All Models)
# ---------------------------------------------------------------------------


def chart_03_compound_learning_curves(study_data, labels, output_dir):
    """
    All models' validation accuracy curves overlaid on a single axes.

    Saves: output_dir/03_compound_learning_curves.png
    """
    plotted = False
    global_min = 100.0

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (datum, label) in enumerate(zip(study_data, labels)):
            df = datum["metrics"]
            if df is None:
                print(f"[Chart 03] Skipping '{label}' — no metrics.csv")
                continue

            epochs = df["epoch"]
            acc = df["val_acc_top1"] * 100
            color = COLORS[i % len(COLORS)]

            ax.plot(epochs, acc, color=color, linewidth=2, label=label)

            # Mark best epoch with a dot
            best_epoch = datum["results"].get("best_epoch")
            if best_epoch is not None and best_epoch in epochs.values:
                best_mask = epochs == best_epoch
                best_acc = acc[best_mask].values[0]
                ax.scatter(best_epoch, best_acc, color=color, s=60,
                           zorder=5, marker="o")

            global_min = min(global_min, acc.min())
            plotted = True

        if not plotted:
            print("[Chart 03] No data available — skipping")
            plt.close(fig)
            return

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.set_title("Validation Accuracy Learning Curves")
        ax.legend(loc="lower right")
        ax.set_ylim(max(0, global_min - 5), 101)

        plt.tight_layout()
        save_path = output_dir / "03_compound_learning_curves.png"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Chart 03] Saved: {save_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4 — Per-Class Compound Learning Curves (top-2 models)
# ---------------------------------------------------------------------------


def chart_04_class_compound_curves(study_data, labels, output_dir):
    """
    2×2 compound chart with one subplot per class.  Each subplot overlays the
    val_f1_{class} learning curves for the top-2 models ranked by
    final_test_acc_top1, with a marker at each model's best epoch.

    Saves: output_dir/04_per_class_compound_curves.png
    """
    # Rank studies that have both metrics and a valid accuracy score
    ranked = sorted(
        [
            (datum, label)
            for datum, label in zip(study_data, labels)
            if datum["metrics"] is not None
            and datum["results"].get("final_test_acc_top1") is not None
        ],
        key=lambda x: x[0]["results"]["final_test_acc_top1"],
        reverse=True,
    )

    if not ranked:
        print("[Chart 04] No studies with metrics — skipping")
        return

    top2 = ranked[:2]

    # Discover class names from val_f1_* columns of the first available study
    all_classes = []
    for datum, _ in top2:
        df = datum["metrics"]
        for col in df.columns:
            if col.startswith("val_f1_"):
                cls = col[len("val_f1_"):]
                if cls not in all_classes:
                    all_classes.append(cls)
        if all_classes:
            break
    all_classes = sorted(all_classes)

    if not all_classes:
        print("[Chart 04] No val_f1_<class> columns found — skipping")
        return

    n_classes = len(all_classes)
    n_cols_grid = 2
    n_rows_grid = (n_classes + 1) // 2  # ceiling division

    top2_labels = [lbl for _, lbl in top2]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            n_rows_grid, n_cols_grid,
            figsize=(11, 4 * n_rows_grid),
            sharex=False,
        )
        axes = np.array(axes).flatten()

        for idx, cls in enumerate(all_classes):
            ax = axes[idx]
            col = f"val_f1_{cls}"

            for model_idx, (datum, lbl) in enumerate(top2):
                df = datum["metrics"]
                if col not in df.columns:
                    continue
                epochs = df["epoch"]
                values = df[col] * 100
                color = COLORS[model_idx % len(COLORS)]

                ax.plot(epochs, values, color=color, linewidth=1.5, label=lbl)

                # Mark the best epoch with a filled dot
                best_epoch = datum["results"].get("best_epoch")
                if best_epoch is not None and best_epoch in epochs.values:
                    mask = epochs == best_epoch
                    best_val = values[mask].values[0]
                    ax.scatter(best_epoch, best_val, color=color, s=40,
                               zorder=5, marker="o")

            ax.set_title(cls.replace("_", " ").title())
            ax.set_ylabel("Validation F1 (%)")
            # Only label x-axis on the bottom row
            if idx >= n_classes - n_cols_grid:
                ax.set_xlabel("Epoch")

        # Hide any unused subplot panels
        for idx in range(n_classes, len(axes)):
            axes[idx].set_visible(False)

        # Shared legend above all subplots
        handles, leg_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, leg_labels,
            loc="upper center", ncol=len(top2),
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
        )
        fig.suptitle(
            "Per-Class Validation F1 — Top-2 Models",
            fontsize=11, fontweight="bold", y=1.06,
        )

        plt.tight_layout()
        save_path = output_dir / "04_per_class_compound_curves.png"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Chart 04] Saved: {save_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 5 — Per-Class Performance Bar Chart (all models)
# ---------------------------------------------------------------------------


def chart_05_per_class_performance(study_data, labels, output_dir):
    """
    Grouped bar chart of per-class test accuracy and F1 across all models.
    Two stacked subplots: top = accuracy, bottom = F1.

    Saves: output_dir/05_per_class_performance.png
    """
    # Filter to studies that have per-class data
    valid = []
    for datum, label in zip(study_data, labels):
        has_acc = "per_class_test_acc" in datum["results"]
        has_f1 = "per_class_test_f1" in datum["results"]
        if has_acc or has_f1:
            valid.append((datum, label))
        else:
            print(f"[Chart 05] Skipping '{label}' — missing per_class_test_acc/f1 "
                  f"(re-run training with updated trainer.py)")

    if not valid:
        print("[Chart 05] No per-class test data available — skipping")
        return

    # Collect union of all class names
    all_classes = []
    for datum, _ in valid:
        for key in ("per_class_test_acc", "per_class_test_f1"):
            if key in datum["results"]:
                for cls in datum["results"][key]:
                    if cls not in all_classes:
                        all_classes.append(cls)
    all_classes = sorted(all_classes)
    n_classes = len(all_classes)
    n_models = len(valid)

    # Build matrices (models × classes)
    acc_matrix = np.full((n_models, n_classes), np.nan)
    f1_matrix = np.full((n_models, n_classes), np.nan)
    for i, (datum, _) in enumerate(valid):
        for j, cls in enumerate(all_classes):
            if "per_class_test_acc" in datum["results"]:
                acc_matrix[i, j] = datum["results"]["per_class_test_acc"].get(cls, np.nan)
            if "per_class_test_f1" in datum["results"]:
                f1_matrix[i, j] = datum["results"]["per_class_test_f1"].get(cls, np.nan)

    # Compute a dynamic y-axis floor: round the minimum value down to the
    # nearest 0.05, then subtract 0.05 as a visual buffer.
    all_vals = np.concatenate([acc_matrix.flatten(), f1_matrix.flatten()])
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) > 0:
        y_floor = max(0.0, np.floor(all_vals.min() * 20) / 20 - 0.05)
    else:
        y_floor = 0.0

    bar_width = 0.8 / n_models
    x = np.arange(n_classes)
    fig_w = max(10, n_classes * 1.8)
    valid_labels = [lbl for _, lbl in valid]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(fig_w, 10), sharex=True)

        for subplot_idx, (matrix, title, ylabel) in enumerate([
            (acc_matrix, "Per-Class Test Accuracy", "Accuracy"),
            (f1_matrix,  "Per-Class Test F1",       "F1 Score"),
        ]):
            ax = axes[subplot_idx]
            for i in range(n_models):
                offsets = x + (i - n_models / 2 + 0.5) * bar_width
                vals = matrix[i]
                ax.bar(offsets, np.nan_to_num(vals), width=bar_width,
                       color=COLORS[i % len(COLORS)],
                       label=valid_labels[i] if subplot_idx == 0 else "_nolegend_")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(y_floor, 1.05)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(
            [c.replace("_", " ").title() for c in all_classes],
            rotation=45, ha="right",
        )

        # Shared legend above both subplots
        handles, leg_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, leg_labels, loc="upper center",
                   ncol=min(n_models, 4), fontsize=9,
                   bbox_to_anchor=(0.5, 1.01))
        fig.suptitle("Per-Class Test Performance", fontsize=13,
                     fontweight="bold", y=1.04)

        plt.tight_layout()
        save_path = output_dir / "05_per_class_performance.png"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Chart 05] Saved: {save_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 6 — Summary Table (PNG + CSV)
# ---------------------------------------------------------------------------


def chart_07_summary_table(study_data, labels, output_dir):
    """
    Publication-quality summary table saved as PNG and CSV.

    Rows: one per study. Columns: Model, Test Acc, Macro F1, Precision,
    Recall, Params (M), Inference (ms), per-class F1 (one col per class).

    Saves: output_dir/07_summary_table.png
           output_dir/07_summary_table.csv
    """
    # Collect union of class names from per_class_test_f1 across all studies
    all_classes = []
    for datum in study_data:
        pc = datum["results"].get("per_class_test_f1", {})
        for cls in pc:
            if cls not in all_classes:
                all_classes.append(cls)
    all_classes = sorted(all_classes)

    # Build rows
    fixed_cols = [
        "Model", "Test Acc (%)", "Macro F1", "Precision", "Recall",
        "Params (M)", "Inference (ms)",
    ]
    class_cols = [f"F1 ({c.replace('_', ' ').title()})" for c in all_classes]
    columns = fixed_cols + class_cols

    rows = []
    for datum, label in zip(study_data, labels):
        r = datum["results"]

        acc = r.get("final_test_acc_top1")
        f1 = r.get("final_test_f1")
        prec = r.get("final_test_precision")
        rec = r.get("final_test_recall")
        params = r.get("trainable_params")
        infer = r.get("inference_time_ms")
        pc_f1 = r.get("per_class_test_f1", {})

        row = [
            label,
            f"{acc*100:.2f}" if acc is not None else "N/A",
            f"{f1:.4f}"   if f1   is not None else "N/A",
            f"{prec:.4f}" if prec is not None else "N/A",
            f"{rec:.4f}"  if rec  is not None else "N/A",
            f"{params/1e6:.2f}" if params is not None else "N/A",
            f"{infer:.2f}"      if infer  is not None else "N/A",
        ]
        for cls in all_classes:
            v = pc_f1.get(cls)
            row.append(f"{v:.4f}" if v is not None else "N/A")

        rows.append(row)

    # --- CSV ---
    df = pd.DataFrame(rows, columns=columns)
    csv_path = output_dir / "07_summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Chart 07] Saved: {csv_path}")

    # --- PNG ---
    n_rows = len(rows)
    n_cols = len(columns)
    fig_w = max(14, n_cols * 1.4)
    fig_h = max(4, n_rows * 0.7 + 1.5)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        col_widths = [0.18] + [0.11] * (n_cols - 1)
        table = ax.table(
            cellText=rows,
            colLabels=columns,
            cellLoc="center",
            loc="center",
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.8)

        # Header styling
        for col_idx in range(n_cols):
            cell = table[0, col_idx]
            cell.set_facecolor("#404040")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")

        # Row alternating colors
        for row_idx in range(1, n_rows + 1):
            bg = "#F0F0F0" if row_idx % 2 == 0 else "white"
            for col_idx in range(n_cols):
                table[row_idx, col_idx].set_facecolor(bg)

        ax.set_title("Summary of Model Performance",
                     fontsize=12, fontweight="bold", pad=20)

        plt.tight_layout()
        save_path = output_dir / "07_summary_table.png"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Chart 07] Saved: {save_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_all_charts(study_names, labels, output_dir, study_dir):
    """
    Load all study data and generate all 6 chart types.

    Args:
        study_names: list of Optuna study directory names
        labels:      display names (None → default to backbone names)
        output_dir:  destination for all generated files
        study_dir:   base path containing study subdirectories
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_dir = Path(study_dir)

    print(f"\n[Charts] Loading {len(study_names)} study/studies...")
    study_data = [load_study_best(name, study_dir) for name in study_names]

    # Default labels from backbone names extracted from config.json
    if labels is None:
        labels = [d["backbone"] for d in study_data]

    print(f"[Charts] Output directory: {output_dir}\n")

    # Chart 1 — Accuracy bar
    chart_01_accuracy_bar(study_data, labels, output_dir)

    # Chart 2 — Bubble (size vs accuracy)
    chart_02_bubble_size_accuracy(study_data, labels, output_dir)

    # Chart 3 — Compound learning curves
    chart_03_compound_learning_curves(study_data, labels, output_dir)

    # Chart 4 — Per-class compound learning curves (top-2 models, 2×2 grid)
    chart_04_class_compound_curves(study_data, labels, output_dir)

    # Chart 5 — Per-class performance bar chart
    chart_05_per_class_performance(study_data, labels, output_dir)

    # Chart 6 — Summary table (PNG + CSV)
    chart_07_summary_table(study_data, labels, output_dir)

    print(f"\n[Charts] Done. All figures saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-quality figures from Optuna study results.\n"
            "Reads from results/optuna_studies/{study_name}/best/ by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_paper_charts.py resnet152_fine_tuning_sweep\n"
            "  python generate_paper_charts.py study_a study_b "
            "--labels \"ResNet-152\" \"EfficientNet-B4\"\n"
        ),
    )
    parser.add_argument(
        "studies",
        nargs="+",
        metavar="STUDY",
        help="One or more Optuna study names (directory names under --study-dir)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        default=None,
        help="Display name for each study (must match count of STUDY args). "
             "Defaults to backbone name from each study's config.json.",
    )
    parser.add_argument(
        "--output",
        default="results/paper_figures",
        metavar="DIR",
        help="Output directory for generated figures (default: results/paper_figures)",
    )
    parser.add_argument(
        "--study-dir",
        default="results/optuna_studies",
        metavar="DIR",
        help="Base directory containing study subdirectories "
             "(default: results/optuna_studies)",
    )

    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.studies):
        parser.error(
            f"--labels count ({len(args.labels)}) must match "
            f"study count ({len(args.studies)})"
        )

    return args


def main():
    args = parse_args()
    generate_all_charts(
        study_names=args.studies,
        labels=args.labels,
        output_dir=args.output,
        study_dir=args.study_dir,
    )


if __name__ == "__main__":
    main()
