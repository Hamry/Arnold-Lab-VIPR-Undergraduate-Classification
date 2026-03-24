"""
audit_dataset.py — Evaluate a trained model on a data split and save misclassified images.

Usage:
    python audit_dataset.py results/optuna_studies/resnet152_fine_tuning_sweep/best
    python audit_dataset.py results/resnet152_frozen --split test --output my_audit
    python audit_dataset.py results/optuna_studies/.../best --data-path /local/path/to/data

Output:
    dataset_audit/
        {CorrectClass}_as_{GuessedClass}/
            {XXXX}_{filename}   ← confidence as basis points (9421 = 94.21%)
        audit_summary.txt
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# Make sure the project root is on the path so utils imports work
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_utils import load_model
from utils.trainer import build_eval_transforms, load_checkpoint, logits_to_probs


def parse_args():
    parser = argparse.ArgumentParser(description="Audit dataset misclassifications")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to experiment results directory containing config.json and best_model.pth",
    )
    parser.add_argument(
        "--split",
        default="validate",
        choices=["train", "validate", "test"],
        help="Dataset split to evaluate (default: validate)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_audit"),
        help="Output directory for misclassified images (default: dataset_audit)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override the data path from config (useful when running locally vs. server)",
    )
    return parser.parse_args()


def load_config(results_dir: Path, data_path_override: Path | None) -> dict:
    config_path = results_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {results_dir}")
    with open(config_path) as f:
        options = json.load(f)
    if data_path_override is not None:
        options["data"]["path"] = str(data_path_override)
    return options


def run_inference(model, dataset, batch_size: int, device: torch.device,
                  loss_fn: str = "cross_entropy"):
    """Run inference over a dataset, returning per-image targets, preds, confidences, paths."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    all_targets = []
    all_preds = []
    all_confs = []
    all_paths = []

    global_idx = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = logits_to_probs(outputs, loss_fn)
            confs, preds = probs.max(dim=1)

            n = len(labels)
            batch_paths = [dataset.samples[global_idx + i][0] for i in range(n)]
            global_idx += n

            all_targets.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())
            all_paths.extend(batch_paths)

    return all_targets, all_preds, all_confs, all_paths


def copy_misclassified(targets, preds, confs, paths, class_names, output_dir: Path):
    """Copy misclassified images into output_dir/{Correct}_as_{Guessed}/ folders."""
    bucket_counts: dict[str, int] = {}

    for target, pred, conf, img_path in zip(targets, preds, confs, paths):
        if pred == target:
            continue

        correct = class_names[target]
        guessed = class_names[pred]
        bucket = f"{correct}_as_{guessed}"

        folder = output_dir / bucket
        folder.mkdir(parents=True, exist_ok=True)

        conf_prefix = f"{int(conf * 10000):04d}"
        orig_name = Path(img_path).name
        dest_name = f"{conf_prefix}_{orig_name}"
        dest = folder / dest_name

        shutil.copy2(img_path, dest)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    return bucket_counts


def compute_per_class_accuracy(targets, preds, class_names):
    counts = {name: {"correct": 0, "total": 0} for name in class_names}
    for target, pred in zip(targets, preds):
        name = class_names[target]
        counts[name]["total"] += 1
        if pred == target:
            counts[name]["correct"] += 1
    return {
        name: (v["correct"] / v["total"] if v["total"] > 0 else float("nan"))
        for name, v in counts.items()
    }


def build_summary(
    split: str,
    results_dir: Path,
    class_names: list[str],
    targets: list,
    preds: list,
    bucket_counts: dict,
) -> str:
    total = len(targets)
    correct = sum(t == p for t, p in zip(targets, preds))
    overall_acc = correct / total if total else 0.0
    per_class_acc = compute_per_class_accuracy(targets, preds, class_names)

    lines = [
        "=" * 60,
        f"Dataset Audit Summary",
        f"  Results dir : {results_dir}",
        f"  Split       : {split}",
        f"  Total images: {total}",
        f"  Correct     : {correct}",
        f"  Overall acc : {overall_acc:.4f} ({overall_acc * 100:.2f}%)",
        "=" * 60,
        "",
        "Per-class accuracy:",
    ]
    for name, acc in per_class_acc.items():
        lines.append(f"  {name:<12} {acc:.4f} ({acc * 100:.2f}%)")

    lines += ["", "Misclassification buckets (correct_as_guessed : count):"]
    if bucket_counts:
        for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {bucket:<30} {count}")
    else:
        lines.append("  None — perfect classification!")

    lines += ["", f"Total misclassified: {total - correct}"]
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    args = parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"[Error] Results directory not found: {results_dir}")
        sys.exit(1)

    checkpoint_path = results_dir / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"[Error] best_model.pth not found in {results_dir}")
        sys.exit(1)

    print(f"[Audit] Loading config from {results_dir / 'config.json'}")
    options = load_config(results_dir, args.data_path)

    data_path = Path(options["data"]["path"])
    split_path = data_path / args.split
    if not split_path.exists():
        print(f"[Error] Split path not found: {split_path}")
        print(f"        Use --data-path to override the data root from config.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Audit] Device: {device}")

    print(f"[Audit] Building model ({options['model']['backbone']}) ...")
    model = load_model(options, num_classes=4)
    load_checkpoint(model, results_dir)
    model = model.to(device)
    model.eval()

    print(f"[Audit] Loading '{args.split}' split from {split_path}")
    transform = build_eval_transforms(options)
    dataset = datasets.ImageFolder(str(split_path), transform=transform)
    class_names = dataset.classes
    print(f"[Audit] Classes: {class_names}  |  Images: {len(dataset)}")

    loss_fn = options["model"].get("loss_fn", "cross_entropy")
    batch_size = options["data"].get("batch_size", 32)
    print(f"[Audit] Running inference (batch_size={batch_size}) ...")
    targets, preds, confs, paths = run_inference(model, dataset, batch_size, device,
                                                 loss_fn=loss_fn)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Audit] Copying misclassified images to {output_dir} ...")
    bucket_counts = copy_misclassified(
        targets, preds, confs, paths, class_names, output_dir
    )

    summary = build_summary(
        args.split, results_dir, class_names, targets, preds, bucket_counts
    )
    print()
    print(summary)

    summary_path = output_dir / "audit_summary.txt"
    summary_path.write_text(summary)
    print(f"\n[Audit] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
