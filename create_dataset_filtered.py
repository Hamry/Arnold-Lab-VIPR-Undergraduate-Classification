import csv
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

from utils.model_utils import load_model
from utils.trainer import build_eval_transforms, load_checkpoint


# ---------------------------------------------------------------------------
# Lightweight dataset for inference over a flat list of paths
# ---------------------------------------------------------------------------


class _PathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def filter_and_split_dataset(
    source_dir1,
    source_dir2,
    dest_dir,
    results_dir,
    confidence_threshold=0.7,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    n=1000,
    batch_size=32,
):
    """
    Like merge_and_split_dataset, but each candidate image is first run through
    a trained model.  Images that are *confidently incorrect*
    (predicted class != true class AND softmax confidence >= confidence_threshold)
    are excluded.  Excluded images are logged to {dest_dir}/excluded_images.csv.

    Args:
        source_dir1 (str): Path to the first source directory (e.g., 'colonized').
        source_dir2 (str): Path to the second source directory (e.g., 'noncolonized').
        dest_dir (str): Path to the destination directory for the split dataset.
        results_dir (str): Path to the experiment results directory containing
                           config.json and best_model.pth.
        confidence_threshold (float): Confidence above which a wrong prediction
                                      causes the image to be excluded (default 0.7).
        train_ratio (float): Proportion for the training set.
        val_ratio (float): Proportion for the validation set.
        test_ratio (float): Proportion for the test set.
        n (int): Target number of images per category (default 1000).
        batch_size (int): Batch size for inference (default 32).
    """
    # --- Validation ---
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        print("Error: Ratios must sum to 1.0")
        return

    categories = ["Blurry", "Good", "Opaque", "Yellow"]
    # Class index mapping matches torchvision ImageFolder alphabetical order:
    #   Blurry=0, Good=1, Opaque=2, Yellow=3
    class_to_idx = {c: i for i, c in enumerate(sorted(categories))}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    splits = ["train", "validate", "test"]

    results_dir = Path(results_dir)
    dest_dir = Path(dest_dir)

    # --- Load model ---
    config_path = results_dir / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {results_dir}")
        return
    with open(config_path) as f:
        options = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model ({options['model']['backbone']}) on {device} ...")
    model = load_model(options, num_classes=len(categories))
    load_checkpoint(model, results_dir)
    model = model.to(device)
    model.eval()

    transform = build_eval_transforms(options)

    # --- Create destination directory structure ---
    print(f"Setting up destination directory at: {dest_dir}")
    for split in splits:
        for category in categories:
            (dest_dir / split / category).mkdir(parents=True, exist_ok=True)

    # --- Pre-scan: build {source_dir -> {category -> set_of_basenames}} ---
    print("\nPre-scanning source directories for cross-category deduplication...")
    source_basenames = {}
    for source_main_dir in [source_dir1, source_dir2]:
        cat_to_names = {}
        for cat in categories:
            cat_path = os.path.join(source_main_dir, cat)
            if os.path.isdir(cat_path):
                cat_to_names[cat] = {
                    f for f in os.listdir(cat_path) if f.lower().endswith(".png")
                }
            else:
                cat_to_names[cat] = set()
        source_basenames[source_main_dir] = cat_to_names

    # --- Collect exclusion log entries across all categories ---
    all_excluded_rows = []  # list of (image_path, true_class, pred_class, confidence)

    for category in categories:
        true_label = class_to_idx[category]
        print(f"\nProcessing category: {category}")

        # --- Gather candidates (same dedup logic as create_dataset.py) ---
        candidate_paths = []
        seen_cross_source = set()

        for source_main_dir in [source_dir1, source_dir2]:
            category_path = os.path.join(source_main_dir, category)
            if not os.path.isdir(category_path):
                print(f"  Warning: Directory not found, skipping: {category_path}")
                continue

            exclusion_set = set()
            for other_cat, other_names in source_basenames[source_main_dir].items():
                if other_cat != category:
                    exclusion_set |= other_names
            exclusion_set |= seen_cross_source

            all_files = [
                f for f in os.listdir(category_path) if f.lower().endswith(".png")
            ]
            excluded_dedup = [f for f in all_files if f in exclusion_set]
            kept = [f for f in all_files if f not in exclusion_set]

            if excluded_dedup:
                print(
                    f"  [{source_main_dir}] Excluded {len(excluded_dedup)} file(s) from"
                    f" '{category}' (duplicate across category or source directory)."
                )

            images = [os.path.join(category_path, f) for f in kept]
            random.shuffle(images)
            images = images[: n // 2]
            candidate_paths.extend(images)
            seen_cross_source.update(os.path.basename(p) for p in images)

        print(f"  Candidate images before model filtering: {len(candidate_paths)}")

        # --- Run inference on all candidates ---
        dataset = _PathDataset(candidate_paths, transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        results = {}  # idx -> (pred_class_name, confidence, keep)
        with torch.no_grad():
            for images_batch, indices in loader:
                images_batch = images_batch.to(device)
                outputs = model(images_batch)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = probs.max(dim=1)
                for i, (pred, conf) in enumerate(
                    zip(preds.cpu().tolist(), confs.cpu().tolist())
                ):
                    idx = indices[i].item()
                    confidently_wrong = (pred != true_label) and (
                        conf >= confidence_threshold
                    )
                    results[idx] = (pred, conf, not confidently_wrong)

        # --- Partition into kept / excluded ---
        kept_paths = []
        excluded_count = 0
        for idx, path in enumerate(candidate_paths):
            pred, conf, keep = results[idx]
            if keep:
                kept_paths.append(path)
            else:
                excluded_count += 1
                all_excluded_rows.append(
                    (path, category, idx_to_class[pred], f"{conf:.4f}")
                )

        print(
            f"  Model excluded {excluded_count} image(s) (confidently incorrect, threshold={confidence_threshold})."
        )
        print(f"  Remaining after model filter: {len(kept_paths)}")

        # --- Shuffle and split ---
        random.shuffle(kept_paths)
        total_images = len(kept_paths)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)

        train_files = kept_paths[:train_end]
        val_files = kept_paths[train_end:val_end]
        test_files = kept_paths[val_end:]

        print(
            f"  Splitting into: {len(train_files)} train,"
            f" {len(val_files)} validate, {len(test_files)} test."
        )

        def copy_files(files, split_name):
            for source_path in files:
                filename = os.path.basename(source_path)
                dest_path = dest_dir / split_name / category / filename
                shutil.copy2(source_path, dest_path)

        copy_files(train_files, "train")
        copy_files(val_files, "validate")
        copy_files(test_files, "test")

    # --- Write exclusion log ---
    if all_excluded_rows:
        log_path = dest_dir / "excluded_images.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["image_path", "true_class", "predicted_class", "confidence"]
            )
            writer.writerows(all_excluded_rows)
        print(
            f"\nExclusion log written to: {log_path}  ({len(all_excluded_rows)} images excluded total)"
        )
    else:
        print("\nNo images were excluded by the model.")

    print("\nDataset filtering and splitting complete!")


# =============================================================================
# --- HOW TO USE ---
# =============================================================================
# 0. Define the base path where your source and destination folders live.
base_path = "/work/ompla/HL98745"

# 1. Define the path to your 'colonized' images folder.
colonized_folder = f"{base_path}/Colonized"

# 2. Define the path to your 'noncolonized' images folder.
noncolonized_folder = f"{base_path}/Noncolonized"

# 3. Define where you want the final, split dataset to be created.
final_dataset_folder = f"{base_path}/data/single_label_category_balanced_filtered/"

# 4. Define the path to your experiment results directory.
#    This folder must contain config.json and best_model.pth.
model_results_dir = "./results/optuna_studies/convnext_base_frozen_sweep"

# 5. Run the function with your defined paths.
filter_and_split_dataset(
    source_dir1=colonized_folder,
    source_dir2=noncolonized_folder,
    dest_dir=final_dataset_folder,
    results_dir=model_results_dir,
    confidence_threshold=0.7,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    n=1000,
)
