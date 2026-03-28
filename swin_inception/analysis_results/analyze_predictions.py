"""
analyze_predictions.py — summarize inference CSVs from both models.

Usage:
    python analyze_predictions.py
    python analyze_predictions.py --inception predictions_inception.csv \
                                  --swin      predictions_swin.csv \
                                  --threshold 0.90 \
                                  --outdir    analysis_results
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CLASS_NAMES = ["Blurry", "Good", "Opaque", "Yellow"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["predicted"] = df[CLASS_NAMES].idxmax(axis=1)
    df["confidence"] = df[CLASS_NAMES].max(axis=1)
    return df


def class_frequency(df: pd.DataFrame, label: str):
    counts = df["predicted"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    pcts   = counts / len(df) * 100
    print(f"\n{'='*55}")
    print(f"  Class frequency — {label}  (n={len(df):,})")
    print(f"{'='*55}")
    for cls in CLASS_NAMES:
        bar = "█" * int(pcts[cls] / 2)
        print(f"  {cls:8s}  {counts[cls]:6,}  ({pcts[cls]:5.1f}%)  {bar}")
    return counts, pcts


def corrected_frequency(df: pd.DataFrame, label: str,
                        acc: dict = None):
    """
    Naive correction: predicted_count / model_precision_for_that_class.
    acc = {"Blurry": 0.91, "Good": 0.89, "Opaque": 0.90, "Yellow": 0.93}
    These are the precision values from the final training report.
    """
    if acc is None:
        # Default: Inception V3 final precision values from training log
        acc = {"Blurry": 0.914, "Good": 0.886,
               "Opaque": 0.895, "Yellow": 0.935}

    counts = df["predicted"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    print(f"\n{'='*55}")
    print(f"  Corrected class frequency — {label}")
    print(f"  (predicted count / model precision)")
    print(f"{'='*55}")
    for cls in CLASS_NAMES:
        corrected = counts[cls] / acc[cls]
        print(f"  {cls:8s}  raw={counts[cls]:6,}  "
              f"corrected≈{corrected:7,.0f}  (÷{acc[cls]:.3f})")


def high_confidence_good(df: pd.DataFrame, threshold: float,
                         out_path: str):
    """Save filenames of high-confidence Good images — Henry's clean dataset."""
    good_df = df[(df["predicted"] == "Good") & (df["Good"] >= threshold)]
    good_df[["filename", "Good"]].to_csv(out_path, index=False)
    print(f"\n  High-confidence Good (≥{threshold:.0%}): "
          f"{len(good_df):,} images → {out_path}")
    return good_df


def agreement(df_inc: pd.DataFrame, df_swin: pd.DataFrame):
    """How often do both models agree on the predicted class?"""
    merged = df_inc[["filename", "predicted"]].merge(
        df_swin[["filename", "predicted"]],
        on="filename", suffixes=("_inc", "_swin")
    )
    agree = (merged["predicted_inc"] == merged["predicted_swin"]).sum()
    pct   = agree / len(merged) * 100
    print(f"\n{'='*55}")
    print(f"  Model agreement on {len(merged):,} shared images")
    print(f"  {'Agree':8s}  {agree:6,}  ({pct:.1f}%)")
    print(f"  {'Disagree':8s}  {len(merged)-agree:6,}  ({100-pct:.1f}%)")

    # Per-class agreement
    print(f"\n  Per-class agreement:")
    for cls in CLASS_NAMES:
        sub = merged[merged["predicted_inc"] == cls]
        if len(sub) == 0:
            continue
        cls_agree = (sub["predicted_swin"] == cls).sum()
        print(f"    {cls:8s}  {cls_agree}/{len(sub)} "
              f"({100*cls_agree/len(sub):.1f}%)")
    return merged


def plot_distributions(df_inc: pd.DataFrame, df_swin: pd.DataFrame,
                       out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    colors = ["#E24B4A", "#1D9E75", "#BA7517", "#378ADD"]

    for ax, df, title in zip(
        axes,
        [df_inc, df_swin],
        ["Inception V3", "Swin Transformer"]
    ):
        counts = df["predicted"].value_counts().reindex(CLASS_NAMES, fill_value=0)
        bars   = ax.bar(CLASS_NAMES, counts.values, color=colors, width=0.6)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Image count")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"
        ))
        ax.set_ylim(0, counts.max() * 1.18)
        ax.spines[["top", "right"]].set_visible(False)

        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + counts.max() * 0.01,
                    f"{val:,}", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Predicted class distribution — SAM dataset (n=25,611)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {out}")


def plot_confidence(df_inc: pd.DataFrame, df_swin: pd.DataFrame,
                    out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, df, title in zip(
        axes,
        [df_inc, df_swin],
        ["Inception V3", "Swin Transformer"]
    ):
        ax.hist(df["confidence"], bins=40, color="#185FA5", edgecolor="white",
                linewidth=0.4)
        ax.axvline(0.9, color="#E24B4A", linestyle="--", linewidth=1.2,
                   label="90% threshold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Max class probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Prediction confidence distribution", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(out_dir, "confidence_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inception",  default="predictions_inception.csv")
    p.add_argument("--swin",       default="predictions_swin.csv")
    p.add_argument("--threshold",  type=float, default=0.90,
                   help="Confidence threshold for 'good' dataset extraction")
    p.add_argument("--outdir",     default="analysis_results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading CSVs…")
    df_inc  = load_predictions(args.inception)
    df_swin = load_predictions(args.swin)

    # 1. Class frequencies
    class_frequency(df_inc,  "Inception V3")
    class_frequency(df_swin, "Swin Transformer")

    # 2. Corrected (true) class frequency estimates
    corrected_frequency(df_inc, "Inception V3")
    corrected_frequency(
        df_swin, "Swin Transformer",
        acc={"Blurry": 0.858, "Good": 0.894,
             "Opaque": 0.918, "Yellow": 0.879}  # Swin final precision values
    )

    # 3. High-confidence Good images (clean dataset candidates)
    high_confidence_good(
        df_inc, args.threshold,
        os.path.join(args.outdir, "good_candidates_inception.csv")
    )
    high_confidence_good(
        df_swin, args.threshold,
        os.path.join(args.outdir, "good_candidates_swin.csv")
    )

    # 4. Model agreement
    agreement(df_inc, df_swin)

    # 5. Plots
    plot_distributions(df_inc, df_swin, args.outdir)
    plot_confidence(df_inc, df_swin, args.outdir)

    print(f"\nAll outputs written to ./{args.outdir}/")


if __name__ == "__main__":
    main()
