"""
Inception V3 Inference Script — VIPR / MycorrhiSEE Dataset

Usage:
    python inference_inception.py \
        --model results_inception_final/best_model.pth \
        --data /work/omlpa/SAM \
        --output predictions_inception.csv
"""

import argparse
import os
import glob
import csv

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ── constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES = 4
CLASS_NAMES = ["Blurry", "Good", "Opaque", "Yellow"]
INPUT_SIZE  = 299
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── dataset ───────────────────────────────────────────────────────────────────
class PNGDataset(Dataset):
    """Recursively finds all .png files under root_dir."""
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.paths = sorted(
            glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
        )
        if not self.paths:
            raise FileNotFoundError(f"No PNG files found under: {root_dir}")
        print(f"Found {len(self.paths):,} PNG files under {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Could not open {path}: {e} — using blank image")
            img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, path


# ── model builder ─────────────────────────────────────────────────────────────
def infer_hidden_size(state: dict) -> int:
    """
    Read hidden_size directly from the saved fc weights so we never
    have to guess or pass a config file.
    fc.0.weight shape = (hidden_size, backbone_out_features)
    """
    h = state["fc.0.weight"].shape[0]
    print(f"Auto-detected hidden_size={h} from checkpoint weights")
    return h


def build_inception(hidden_size: int, dropout: float = 0.5) -> nn.Module:
    model = models.inception_v3(weights=None)
    nf = model.fc.in_features
    h  = hidden_size
    model.fc = nn.Sequential(
        nn.Linear(nf, h),
        nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h, h // 2),
        nn.BatchNorm1d(h // 2), nn.ReLU(), nn.Dropout(dropout * 0.8),
        nn.Linear(h // 2, NUM_CLASSES),
    )
    if hasattr(model, "AuxLogits"):
        model.AuxLogits.fc = nn.Linear(
            model.AuxLogits.fc.in_features, NUM_CLASSES
        )
    return model


def load_model(pth_path: str) -> nn.Module:
    checkpoint = torch.load(pth_path, map_location=DEVICE, weights_only=False)

    # Unwrap checkpoint dict if needed
    state = checkpoint
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint)

    # Strip DataParallel prefix if present
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    # Detect head shape from the weights themselves — no config file needed
    hidden_size = infer_hidden_size(state)

    model = build_inception(hidden_size=hidden_size)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded {pth_path} → device={DEVICE}")
    return model


# ── inference ─────────────────────────────────────────────────────────────────
def run_inference(model: nn.Module, data_dir: str, out_csv: str,
                  batch_size: int = BATCH_SIZE, use_tta: bool = False):
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = PNGDataset(data_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    out_dir = os.path.dirname(os.path.abspath(out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    written = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + CLASS_NAMES)

        with torch.no_grad():
            for batch_idx, (images, paths) in enumerate(loader):
                images = images.to(DEVICE)

                if use_tta:
                    augmented = [
                        images,
                        torch.flip(images, [3]),
                        torch.flip(images, [2]),
                        torch.flip(images, [2, 3]),
                    ]
                    prob_list = []
                    for aug in augmented:
                        out = model(aug)
                        out = out[0] if isinstance(out, tuple) else out
                        prob_list.append(torch.softmax(out, dim=1))
                    probs = torch.stack(prob_list).mean(0)
                else:
                    out   = model(images)
                    out   = out[0] if isinstance(out, tuple) else out
                    probs = torch.softmax(out, dim=1)

                for path, prob in zip(paths, probs.cpu().numpy()):
                    writer.writerow(
                        [os.path.basename(path)] + [f"{p:.4f}" for p in prob]
                    )
                    written += 1

                if (batch_idx + 1) % 50 == 0:
                    print(f"  {written:,} images processed…")

    print(f"\nDone. {written:,} rows written to {out_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Inception V3 inference on PNG directory"
    )
    p.add_argument("--model",      required=True,
                   help="Path to best_model.pth")
    p.add_argument("--data",       required=True,
                   help="Root directory to scan for .png files (recursive)")
    p.add_argument("--output",     default="predictions_inception.csv",
                   help="Output CSV path")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--tta",        action="store_true",
                   help="Average predictions over 4 flips")
    return p.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    model = load_model(args.model)
    run_inference(model, args.data, args.output,
                  batch_size=args.batch_size, use_tta=args.tta)
