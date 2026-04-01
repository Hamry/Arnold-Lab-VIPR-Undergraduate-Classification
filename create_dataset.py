import os
import shutil
import random
import math


def merge_and_split_dataset(
    source_dir1,
    source_dir2,
    dest_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    n=1000,
):
    """
    Merges two source directories with identical class subfolders and splits them
    into a single new dataset for machine learning.

    Args:
        source_dir1 (str): Path to the first source directory (e.g., 'colonized').
        source_dir2 (str): Path to the second source directory (e.g., 'noncolonized').
        dest_dir (str): Path to the destination directory for the split dataset.
        train_ratio (float): Proportion for the training set.
        val_ratio (float): Proportion for the validation set.
        test_ratio (float): Proportion for the test set.
    """
    # --- 1. Setup and Validation ---
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        print("Error: Ratios must sum to 1.0")
        return

    categories = ["Blurry", "Good", "Opaque", "Yellow"]
    splits = ["train", "validate", "test"]

    # --- 2. Create Destination Directory Structure ---
    print(f"Setting up destination directory at: {dest_dir}")
    for split in splits:
        for category in categories:
            # This creates paths like: '../split_dataset/train/blurry'
            path = os.path.join(dest_dir, split, category)
            os.makedirs(path, exist_ok=True)

    # --- 2b. Pre-scan: Build {source_dir -> {category -> set_of_basenames}} ---
    print("\nPre-scanning source directories for cross-category deduplication...")
    source_basenames = {}
    for source_main_dir in [source_dir1, source_dir2]:
        cat_to_names = {}
        for cat in categories:
            cat_path = os.path.join(source_main_dir, cat)
            if os.path.isdir(cat_path):
                cat_to_names[cat] = {f for f in os.listdir(cat_path) if f.lower().endswith('.png')}
            else:
                cat_to_names[cat] = set()
        source_basenames[source_main_dir] = cat_to_names

    # --- 3. Process Each Category ---
    for category in categories:
        print(f"\nProcessing category: {category}")

        # --- 4. Gather and Combine All Image Paths ---
        all_image_paths = []
        seen_cross_source = set()  # basenames already collected from a prior source dir
        for source_main_dir in [source_dir1, source_dir2]:
            category_path = os.path.join(source_main_dir, category)
            if not os.path.isdir(category_path):
                print(f"Warning: Directory not found, skipping: {category_path}")
                continue

            # Build exclusion set: basenames present in any OTHER category within
            # this source directory, plus basenames already collected from the other
            # source directory for this same category.
            exclusion_set = set()
            for other_cat, other_names in source_basenames[source_main_dir].items():
                if other_cat != category:
                    exclusion_set |= other_names
            exclusion_set |= seen_cross_source

            # Only consider .png files; filter out duplicates, then sample
            all_files = [f for f in os.listdir(category_path) if f.lower().endswith('.png')]
            excluded = [f for f in all_files if f in exclusion_set]
            kept = [f for f in all_files if f not in exclusion_set]
            if excluded:
                print(
                    f"  [{source_main_dir}] Excluded {len(excluded)} file(s) from"
                    f" '{category}' (duplicate across category or source directory)."
                )
            images = [os.path.join(category_path, f) for f in kept]
            random.shuffle(images)
            images = images[: n // 2]
            all_image_paths.extend(images)
            seen_cross_source.update(os.path.basename(p) for p in images)

        # --- 5. Shuffle and Split the Combined List ---
        random.shuffle(all_image_paths)

        total_images = len(all_image_paths)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)

        train_files = all_image_paths[:train_end]
        val_files = all_image_paths[train_end:val_end]
        test_files = all_image_paths[val_end:]

        print(f"  - Found {total_images} total images.")
        print(
            f"  - Splitting into: {len(train_files)} train, {len(val_files)} validate, {len(test_files)} test."
        )

        # --- 6. Copy Files to Their New Homes ---
        def copy_files(files, split_name):
            for source_path in files:
                # Use os.path.basename to get just the filename
                filename = os.path.basename(source_path)
                dest_path = os.path.join(dest_dir, split_name, category, filename)
                shutil.copy2(source_path, dest_path)

        copy_files(train_files, "train")
        copy_files(val_files, "validate")
        copy_files(test_files, "test")

    print("\n✅ Dataset merging and splitting complete!")


# =============================================================================
# --- HOW TO USE ---
# =============================================================================

# 1. Define the path to your 'colonized' images folder.
colonized_folder = "./Colonized"

# 2. Define the path to your 'noncolonized' images folder.
noncolonized_folder = "./Noncolonized"

# 3. Define where you want the final, split dataset to be created.
final_dataset_folder = "./data/single_label_category_balanced/"

# 4. Run the function with your defined paths.
merge_and_split_dataset(
    source_dir1=colonized_folder,
    source_dir2=noncolonized_folder,
    dest_dir=final_dataset_folder,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
)
