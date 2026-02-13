"""
Training Engine Module

Stateless training function that takes a config, runs an experiment,
and saves results. Supports mixed precision and early stopping.
"""

import os
import json
import time
import random
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model_utils import load_model, count_parameters


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def build_train_transforms(options):
    """
    Build training data transforms from config.

    Applies augmentations specified in options['augmentations'],
    then normalizes with ImageNet statistics.
    """
    input_size = options["data"]["input_size"]
    aug = options.get("augmentations", {})

    transform_list = []

    # Random crop with scale
    crop_scale = aug.get("random_crop_scale")
    if crop_scale:
        transform_list.append(
            transforms.RandomResizedCrop(input_size, scale=tuple(crop_scale))
        )
    else:
        transform_list.append(transforms.Resize((input_size, input_size)))

    # Horizontal flip
    if aug.get("horizontal_flip", False):
        transform_list.append(transforms.RandomHorizontalFlip())

    # Vertical flip
    if aug.get("vertical_flip", False):
        transform_list.append(transforms.RandomVerticalFlip())

    # Random rotation
    rotation = aug.get("random_rotation", 0)
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))

    # Color jitter
    jitter = aug.get("color_jitter", 0)
    if jitter > 0:
        transform_list.append(
            transforms.ColorJitter(jitter, jitter, jitter, jitter * 0.5)
        )

    # Standard normalization
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transforms.Compose(transform_list)


def build_eval_transforms(options):
    """
    Build evaluation transforms (no augmentation).

    Resizes slightly larger then center crops to input_size.
    """
    input_size = options["data"]["input_size"]
    resize_size = int(input_size * 1.14)

    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_dataloaders(options):
    """
    Create train, validation, and test dataloaders.

    Expects data path to contain train/, val/, test/ subdirectories.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data_path = Path(options["data"]["path"])
    batch_size = options["data"]["batch_size"]
    num_workers = options["data"].get("num_workers", 4)

    train_transform = build_train_transforms(options)
    eval_transform = build_eval_transforms(options)

    train_dataset = datasets.ImageFolder(data_path / "train", train_transform)
    val_dataset = datasets.ImageFolder(data_path / "validate", eval_transform)
    test_dataset = datasets.ImageFolder(data_path / "test", eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Optimizer and Scheduler
# ---------------------------------------------------------------------------


def create_optimizer(model, options):
    """Create optimizer from config."""
    train_opts = options["training"]
    lr = train_opts["learning_rate"]
    weight_decay = train_opts.get("weight_decay", 0.01)
    optimizer_name = train_opts.get("optimizer", "adamw").lower()

    params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer, options, steps_per_epoch):
    """Create learning rate scheduler from config."""
    train_opts = options["training"]
    scheduler_name = train_opts.get("scheduler")
    epochs = train_opts["epochs"]

    if scheduler_name is None:
        return None
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * steps_per_epoch
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 3, gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


def compute_accuracy(outputs, targets, topk=(1, 3)):
    """
    Compute top-k accuracy for the specified values of k.

    Args:
        outputs: Model logits [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        topk: Tuple of k values to compute

    Returns:
        list: Accuracies for each k value (as floats 0-1)
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k / batch_size)

    return results


# ---------------------------------------------------------------------------
# Training and Evaluation Loops
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    """
    Train for one epoch with mixed precision.

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a dataloader.

    Returns:
        dict: {'loss': float, 'acc_top1': float, 'acc_top3': float}
    """
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            top1, top3 = compute_accuracy(outputs, targets, topk=(1, 3))

            total_loss += loss.item()
            total_top1 += top1.item()
            total_top3 += top3.item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "acc_top1": total_top1 / num_batches,
        "acc_top3": total_top3 / num_batches,
    }


# ---------------------------------------------------------------------------
# Inference Time Measurement
# ---------------------------------------------------------------------------


def measure_inference_time(model, input_size, device, iterations=100):
    """
    Measure average inference time in milliseconds.

    Runs warmup iterations then measures average over specified iterations.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000  # Convert to ms


# ---------------------------------------------------------------------------
# Results Management
# ---------------------------------------------------------------------------


def create_results_dir(experiment_name):
    """Create and return the results directory path."""
    results_dir = Path("results") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_config(options, results_dir):
    """Save a copy of the config to the results directory."""
    config_path = results_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(options, f, indent=2)


def init_metrics_file(results_dir):
    """Initialize the metrics CSV file with headers."""
    metrics_path = results_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_acc_top1", "val_acc_top3", "lr"]
        )
    return metrics_path


def append_metrics(metrics_path, epoch, train_loss, val_metrics, lr):
    """Append one row of metrics to the CSV file."""
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                f"{train_loss:.4f}",
                f"{val_metrics['loss']:.4f}",
                f"{val_metrics['acc_top1']:.4f}",
                f"{val_metrics['acc_top3']:.4f}",
                f"{lr:.6f}",
            ]
        )


def save_checkpoint(model, results_dir, filename="best_model.pth"):
    """Save model weights to the results directory."""
    checkpoint_path = results_dir / filename
    torch.save(model.state_dict(), checkpoint_path)


def load_checkpoint(model, results_dir, filename="best_model.pth"):
    """Load model weights from the results directory."""
    checkpoint_path = results_dir / filename
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """
    Early stopping handler.

    Stops training if validation metric doesn't improve for `patience` epochs.
    """

    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        """
        Update early stopping state.

        Args:
            score: Current validation metric (higher is better)

        Returns:
            bool: True if this is a new best score
        """
        if self.best_score is None:
            self.best_score = score
            return True

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ---------------------------------------------------------------------------
# Console Output
# ---------------------------------------------------------------------------


def print_epoch_summary(epoch, epochs, train_loss, val_metrics, lr, is_best):
    """Print a formatted epoch summary line."""
    best_marker = " *" if is_best else ""
    print(
        f"Epoch {epoch:3d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Top-1: {val_metrics['acc_top1']:.2%} | "
        f"Val Top-3: {val_metrics['acc_top3']:.2%} | "
        f"LR: {lr:.6f}{best_marker}"
    )


def print_final_results(results):
    """Print final experiment results."""
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Validation Accuracy (Top-1): {results['best_val_acc_top1']:.2%}")
    print(f"Best Validation Accuracy (Top-3): {results['best_val_acc_top3']:.2%}")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Test Accuracy (Top-1): {results['final_test_acc_top1']:.2%}")
    print(f"Test Accuracy (Top-3): {results['final_test_acc_top3']:.2%}")
    print(f"Inference Time: {results['inference_time_ms']:.2f} ms")
    print(f"Total Parameters: {results['total_params']:,}")
    print(f"Trainable Parameters: {results['trainable_params']:,}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------


def train_model(options):
    """
    Run a complete training experiment.

    This is the main entry point. Takes a config dict, runs training,
    saves results, and returns a summary.

    Args:
        options: Complete configuration dictionary

    Returns:
        dict: Summary of results including accuracies and timing
    """
    experiment_name = options["experiment_name"]
    epochs = options["training"]["epochs"]
    seed = options["training"].get("seed", 42)
    patience = options["training"].get("early_stopping_patience", 20)

    print(f"\n{'=' * 60}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"{'=' * 60}\n")

    # Setup
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Results directory
    results_dir = create_results_dir(experiment_name)
    save_config(options, results_dir)
    metrics_path = init_metrics_file(results_dir)

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(options)

    # Model
    model = load_model(options)
    model = model.to(device)

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, options)
    scheduler = create_scheduler(optimizer, options, len(train_loader))
    scaler = GradScaler("cuda")

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Training state
    best_val_acc_top1 = 0.0
    best_val_acc_top3 = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )

        val_metrics = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        is_best = early_stopping.step(val_metrics["acc_top1"])

        if is_best:
            best_val_acc_top1 = val_metrics["acc_top1"]
            best_val_acc_top3 = val_metrics["acc_top3"]
            best_epoch = epoch
            save_checkpoint(model, results_dir)

        append_metrics(metrics_path, epoch, train_loss, val_metrics, current_lr)
        print_epoch_summary(epoch, epochs, train_loss, val_metrics, current_lr, is_best)

        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Load best model and evaluate on test set
    load_checkpoint(model, results_dir)
    test_metrics = evaluate(model, test_loader, criterion, device)

    # Measure inference time
    input_size = options["data"]["input_size"]
    inference_time = measure_inference_time(model, input_size, device)

    # Compile results
    total_params, trainable_params = count_parameters(model)

    results = {
        "experiment_name": experiment_name,
        "best_val_acc_top1": best_val_acc_top1,
        "best_val_acc_top3": best_val_acc_top3,
        "best_epoch": best_epoch,
        "final_test_acc_top1": test_metrics["acc_top1"],
        "final_test_acc_top3": test_metrics["acc_top3"],
        "inference_time_ms": inference_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    print_final_results(results)

    # Save results summary
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
