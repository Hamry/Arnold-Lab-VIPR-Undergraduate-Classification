"""
run_inference.py — Unified inference script for VIPR classification models.

Loads any trained model from a results folder (config.json + best_model.pth),
runs batch inference on a directory of PNG images, and produces:
  - predictions_<backbone>.csv
  - class_distribution.png  (with 95% CI bands on corrected counts)
  - confidence_distribution.png

Usage:
    python run_inference.py \
        --result-dir results/optuna_studies/swin_b_frozen_sweep/best \
        --data /path/to/images \
        --output-dir inference_output/swin_b
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add project root to path so utils can be imported when called from subdirs
sys.path.insert(0, str(Path(__file__).parent))
from utils.model_utils import load_model
from utils.trainer import load_checkpoint

# ── constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES = 4
CLASS_NAMES = ["Blurry", "Good", "Opaque", "Yellow"]
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Colorblind-friendly IBM palette (matches utils/visualization.py)
COLORS = ["#648FFF", "#DC267F", "#FFB000", "#FE6100"]

DEFAULT_DATA_PATTERN = "/work/omlpa/SAM/*_data/"


# ── dataset ───────────────────────────────────────────────────────────────────
class PNGDataset(Dataset):
    """Recursively finds all .png files under a directory or glob pattern."""

    def __init__(self, data_pattern: str, input_size: int, transform=None):
        self.input_size = input_size
        self.transform = transform

        matched_roots = sorted(glob.glob(data_pattern))
        if not matched_roots:
            raise FileNotFoundError(f"No directories matched pattern: {data_pattern}")

        all_paths = []
        for root in matched_roots:
            all_paths.extend(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))

        self.paths = sorted(all_paths)
        if not self.paths:
            raise FileNotFoundError(f"No PNG files found under pattern: {data_pattern}")
        print(f"Found {len(self.paths):,} PNG files across {len(matched_roots)} director(ies) matching {data_pattern}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Could not open {path}: {e} — using blank image")
            img = Image.new("RGB", (self.input_size, self.input_size))
        if self.transform:
            img = self.transform(img)
        return img, path


# ── model helpers ─────────────────────────────────────────────────────────────
def load_model_from_result_dir(result_dir: Path):
    """
    Build and load a model from a results folder.

    Reads config.json for architecture, loads best_model.pth for weights.
    Returns (model, backbone_name, input_size).
    """
    config_path = result_dir / "config.json"
    weights_path = result_dir / "best_model.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {result_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"best_model.pth not found in {result_dir}")

    with open(config_path) as f:
        options = json.load(f)

    backbone_name = options["model"]["backbone"]
    input_size = options["data"]["input_size"]
    loss_fn = options["model"].get("loss_fn", "focal_loss_sigmoid")

    print(f"Building {backbone_name} (input={input_size}px) …")
    model = load_model(options)
    load_checkpoint(model, result_dir)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded weights from {weights_path}")
    return model, backbone_name, input_size, loss_fn


def load_class_f1(result_dir: Path) -> dict:
    """
    Return per-class F1 scores keyed by CLASS_NAMES (title-case).
    Falls back to scalar final_test_f1 for all classes if per-class data absent.
    """
    results_path = result_dir / "results.json"
    if not results_path.exists():
        print("[WARN] results.json not found — CI bands will be omitted.")
        return {}

    with open(results_path) as f:
        results = json.load(f)

    per_class = results.get("per_class_test_f1", {})
    if per_class:
        # keys are lowercase ("blurry") — map to CLASS_NAMES titles
        return {cls: per_class.get(cls.lower(), None) for cls in CLASS_NAMES}

    # Fallback: use overall F1 for all classes
    scalar = results.get("final_test_f1") or results.get("final_test_acc_top1")
    if scalar is not None:
        print(
            f"[INFO] No per-class F1 found — using overall F1={scalar:.4f} for all classes."
        )
        return {cls: scalar for cls in CLASS_NAMES}

    return {}


# ── inference ─────────────────────────────────────────────────────────────────
def run_inference(
    model: nn.Module,
    data_dir: str,
    out_csv: str,
    input_size: int,
    loss_fn: str = "focal_loss_sigmoid",
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
):
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = PNGDataset(data_dir, input_size=input_size, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    written = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + CLASS_NAMES)

        with torch.no_grad():
            for batch_idx, (images, paths) in enumerate(loader):
                images = images.to(DEVICE)
                out = model(images)
                # Inception V3 may return (logits, aux_logits) tuple during eval
                if isinstance(out, tuple):
                    out = out[0]
                if loss_fn in ("bce_sigmoid", "focal_loss_sigmoid"):
                    probs = torch.sigmoid(out)
                else:
                    probs = torch.softmax(out, dim=1)

                for path, prob in zip(paths, probs.cpu().numpy()):
                    writer.writerow(
                        [os.path.basename(path)] + [f"{p:.4f}" for p in prob]
                    )
                    written += 1

                if (batch_idx + 1) % 50 == 0:
                    print(f"  {written:,} images processed …")

    print(f"\nDone. {written:,} rows written to {out_csv}")
    return out_csv


# ── analysis helpers ──────────────────────────────────────────────────────────
def load_predictions(csv_path: str):
    """Load predictions CSV, add 'predicted' and 'confidence' columns."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["predicted"] = df[CLASS_NAMES].idxmax(axis=1)
    df["confidence"] = df[CLASS_NAMES].max(axis=1)
    return df


def print_class_frequency(df, label: str):
    counts = df["predicted"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    pcts = counts / len(df) * 100
    print(f"\n{'='*55}")
    print(f"  Class frequency — {label}  (n={len(df):,})")
    print(f"{'='*55}")
    for cls in CLASS_NAMES:
        bar = "█" * int(pcts[cls] / 2)
        print(f"  {cls:8s}  {counts[cls]:6,}  ({pcts[cls]:5.1f}%)  {bar}")
    return counts


def print_corrected_frequency(counts, class_f1: dict, label: str):
    if not class_f1:
        return
    print(f"\n{'='*55}")
    print(f"  Corrected class frequency — {label}")
    print(f"  (predicted count / per-class F1)")
    print(f"{'='*55}")
    for cls in CLASS_NAMES:
        f1 = class_f1.get(cls)
        if f1 and f1 > 0:
            corrected = counts[cls] / f1
            print(
                f"  {cls:8s}  raw={counts[cls]:6,}  corrected≈{corrected:7,.0f}  (÷{f1:.3f})"
            )


def print_high_confidence_good(df, threshold: float = 0.90):
    good_df = df[(df["predicted"] == "Good") & (df["Good"] >= threshold)]
    print(f"\n  High-confidence Good (≥{threshold:.0%}): {len(good_df):,} images")


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_class_distribution(df, class_f1: dict, out_dir: str, backbone_name: str):
    """
    Bar chart of predicted class counts.

    Overlays a diamond marker at the corrected estimate (n_c / f1_c) with
    95% CI error bars using the delta-method approximation:
        μ = n_c / f1_c
        SE = sqrt(n_c * (1 - f1_c)) / f1_c
        95% CI: μ ± 1.96 * SE
    """
    import pandas as pd

    counts = df["predicted"].value_counts().reindex(CLASS_NAMES, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        CLASS_NAMES, counts.values, color=COLORS, width=0.6, label="Predicted count"
    )
    ax.set_ylabel("Image count")
    ax.set_title(
        f"Predicted class distribution — {backbone_name}\n" f"(n={len(df):,} images)",
        fontsize=12,
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.30)
    ax.spines[["top", "right"]].set_visible(False)

    # Label raw counts above bars
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_confidence(df, out_dir: str, backbone_name: str):
    """Histogram of max class probability per image."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
        df["confidence"], bins=40, color="#185FA5", edgecolor="white", linewidth=0.4
    )
    ax.axvline(
        0.9, color="#E24B4A", linestyle="--", linewidth=1.2, label="90% threshold"
    )
    ax.set_title(
        f"Prediction confidence — {backbone_name}", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Max class probability")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(out_dir, "confidence_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_per_class_confidence(df, out_dir: str, backbone_name: str):
    """2×2 grid: confidence distribution for each predicted class."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, cls, color in zip(axes, CLASS_NAMES, COLORS):
        subset = df[df["predicted"] == cls]["confidence"]
        ax.hist(subset, bins=30, color=color, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{cls}  (n={len(subset):,})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Confidence (max softmax prob)")
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Per-class confidence distribution — {backbone_name}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "confidence_per_class.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_opaque_vs_blurry(df, out_dir: str, backbone_name: str):
    """Scatter: Opaque probability vs Blurry probability, colored by predicted class."""
    fig, ax = plt.subplots(figsize=(7, 7))

    color_map = dict(zip(CLASS_NAMES, COLORS))
    for cls in CLASS_NAMES:
        mask = df["predicted"] == cls
        ax.scatter(
            df.loc[mask, "Opaque"],
            df.loc[mask, "Blurry"],
            c=color_map[cls],
            label=cls,
            alpha=0.35,
            s=4,
            linewidths=0,
        )

    # y = x diagonal
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(
        lims,
        lims,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="y = x",
        zorder=5,
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Opaque probability", fontsize=11)
    ax.set_ylabel("Blurry probability", fontsize=11)
    ax.set_title(
        f"Opaque vs Blurry — {backbone_name}\n(n={len(df):,} images)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, markerscale=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    out = os.path.join(out_dir, "opaque_vs_blurry.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference using a trained model from a results folder, "
        "or regenerate plots/analysis from an existing predictions CSV.",
        epilog=(
            "Normal run:  python run_inference.py --result-dir <dir> --data <images/>\n"
            "Regen plots: python run_inference.py --result-dir <dir> --csv predictions_<backbone>.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--result-dir",
        required=True,
        help="Results folder containing config.json, best_model.pth, results.json",
    )
    p.add_argument(
        "--data",
        default=DEFAULT_DATA_PATTERN,
        help=f"Directory or glob pattern to scan for .png files (default: {DEFAULT_DATA_PATTERN})",
    )
    p.add_argument(
        "--csv",
        default=None,
        metavar="PREDICTIONS_CSV",
        help="Path to an existing predictions CSV — skips inference and regenerates plots only",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots (default: <result-dir>/inference_output, "
        "or the CSV's parent directory when --csv is used)",
    )
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--workers", type=int, default=4, help="DataLoader num_workers")
    return p.parse_args()


def run_analysis(df, class_f1: dict, out_dir: str, backbone_name: str):
    """
    Single edit point for all post-inference analysis and plots.

    Add new graphs or calculations here — they will automatically apply
    to both normal inference runs and plot-regeneration runs.
    """
    counts = print_class_frequency(df, backbone_name)
    print_corrected_frequency(counts, class_f1, backbone_name)
    print_high_confidence_good(df)

    print("\nGenerating plots …")
    plot_class_distribution(df, class_f1, out_dir, backbone_name)
    plot_confidence(df, out_dir, backbone_name)
    plot_per_class_confidence(df, out_dir, backbone_name)
    plot_opaque_vs_blurry(df, out_dir, backbone_name)


def main():
    args = parse_args()

    result_dir = Path(args.result_dir).resolve()

    # ── Regeneration mode: skip inference, load existing CSV ──────────────────
    if args.csv:
        csv_path = Path(args.csv).resolve()
        if not csv_path.exists():
            print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
            sys.exit(1)

        out_dir = (
            Path(args.output_dir).resolve() if args.output_dir else csv_path.parent
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        backbone_name = csv_path.stem.replace("predictions_", "")
        print(f"Regenerating plots for {backbone_name} from {csv_path} …")

        class_f1 = load_class_f1(result_dir)
        df = load_predictions(str(csv_path))
        run_analysis(df, class_f1, str(out_dir), backbone_name)
        print(f"\nAll outputs written to {out_dir}")
        return

    # ── Normal inference mode ─────────────────────────────────────────────────
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else result_dir / "inference_output"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    model, backbone_name, input_size, loss_fn = load_model_from_result_dir(result_dir)

    # 2. Load per-class F1 for CI
    class_f1 = load_class_f1(result_dir)

    # 3. Run inference → CSV
    out_csv = str(out_dir / f"predictions_{backbone_name}.csv")
    run_inference(
        model,
        args.data,
        out_csv,
        input_size=input_size,
        loss_fn=loss_fn,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # 4. Analyse and plot
    df = load_predictions(out_csv)
    run_analysis(df, class_f1, str(out_dir), backbone_name)

    print(f"\nAll outputs written to {out_dir}")


if __name__ == "__main__":
    main()
