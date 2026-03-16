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

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SLURM/headless

import torch
import torch.nn as nn
# AMP imports (compatible across PyTorch versions)
try:
    # Newer PyTorch
    from torch.amp import GradScaler, autocast
except Exception:
    # Older / widely supported
    from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model_utils import (
    load_model,
    count_parameters,
    _get_classifier_attr,
    get_backbone_blocks,
    thaw_backbone_percentage,
)
from .visualization import (
    plot_loss_curves,
    plot_accuracy_curves,
    plot_learning_rate,
    plot_experiment_summary,
    plot_confusion_matrix,
)


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


def create_optimizer(model, options, backbone_name=None):
    """
    Create optimizer from config with optional discriminative learning rates.

    When backbone_name is provided and thaw_schedule is configured, creates
    named parameter groups for backbone vs classifier with different LRs.
    Includes ALL parameters (even frozen ones) to preserve momentum state
    when thawing layers mid-training.

    Args:
        model: The model
        options: Config dict
        backbone_name: If provided, enables discriminative LR setup

    Returns:
        torch.optim.Optimizer
    """
    train_opts = options["training"]
    lr = train_opts["learning_rate"]
    weight_decay = train_opts.get("weight_decay", 0.01)
    optimizer_name = train_opts.get("optimizer", "adamw").lower()
    thaw_schedule = train_opts.get("thaw_schedule")

    # Use discriminative LR if thaw_schedule is configured
    if backbone_name and thaw_schedule:
        backbone_lr_ratio = train_opts.get("backbone_lr_ratio", 0.1)
        classifier_attr = _get_classifier_attr(backbone_name)
        classifier = getattr(model, classifier_attr)
        classifier_param_ids = {id(p) for p in classifier.parameters()}

        # Separate ALL params into groups (frozen backbone params included!)
        classifier_params = []
        backbone_params = []
        for param in model.parameters():
            if id(param) in classifier_param_ids:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {'params': classifier_params, 'lr': lr, 'name': 'classifier'},
            {'params': backbone_params, 'lr': lr * backbone_lr_ratio, 'name': 'backbone'},
        ]
    else:
        # Original behavior - only trainable params with single LR
        param_groups = [
            {'params': list(filter(lambda p: p.requires_grad, model.parameters())), 'lr': lr}
        ]

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
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


def check_thaw_schedule(epoch, options):
    """
    Check if unfreezing should occur at this epoch.

    Args:
        epoch: Current epoch number (1-indexed)
        options: Config dict

    Returns:
        float or None: Percentage to unfreeze, or None if no change
    """
    thaw_schedule = options.get("training", {}).get("thaw_schedule")
    if not thaw_schedule:
        return None
    return thaw_schedule.get(str(epoch))


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


def compute_accuracy(outputs, targets):
    """
    Compute top-1 accuracy.

    Args:
        outputs: Model logits [batch_size, num_classes]
        targets: Ground truth labels [batch_size]

    Returns:
        float: Top-1 accuracy (0-1)
    """
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1)).sum()
    return (correct / batch_size).item()


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

        with autocast(enabled=(device.type == "cuda")):
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
        dict: {'loss': float, 'acc_top1': float}
    """
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            top1 = compute_accuracy(outputs, targets)

            total_loss += loss.item()
            total_top1 += top1
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "acc_top1": total_top1 / num_batches,
    }


def evaluate_with_predictions(model, loader, device):
    """
    Evaluate model and collect all predictions for confusion matrix.

    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation
        device: Device to run on

    Returns:
        tuple: (all_targets, all_predictions) as lists
    """
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)

            _, predicted = outputs.max(1)
            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

    return all_targets, all_predictions


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
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc_top1", "lr"])
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
        f"Val Acc: {val_metrics['acc_top1']:.2%} | "
        f"LR: {lr:.6f}{best_marker}"
    )


def print_final_results(results):
    """Print final experiment results."""
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Validation Accuracy: {results['best_val_acc_top1']:.2%}")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Test Accuracy: {results['final_test_acc_top1']:.2%}")
    print(f"Inference Time: {results['inference_time_ms']:.2f} ms")
    print(f"Total Parameters: {results['total_params']:,}")
    print(f"Trainable Parameters: {results['trainable_params']:,}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------


def train_model(options, trial=None, results_dir_override=None):
    """
    Run a complete training experiment.

    This is the main entry point. Takes a config dict, runs training,
    saves results, and returns a summary.

    Args:
        options: Complete configuration dictionary
        trial: Optional Optuna trial for pruning support. If provided,
               reports validation accuracy after each epoch and checks
               for pruning. Raises optuna.TrialPruned if pruned.
        results_dir_override: Optional path to override default results
                             directory. Used by Optuna to place trial
                             results in study-specific directories.

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

    # Results directory (allow override for Optuna)
    if results_dir_override:
        results_dir = Path(results_dir_override)
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = create_results_dir(experiment_name)
    save_config(options, results_dir)
    metrics_path = init_metrics_file(results_dir)

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(options)

    # Model
    model = load_model(options)
    model = model.to(device)
    backbone_name = options['model']['backbone']

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, options, backbone_name)
    scheduler = create_scheduler(optimizer, options, len(train_loader))
    scaler = GradScaler(enabled = (device.type == "cuda"))

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Progressive unfreezing state
    current_thaw_pct = 0.0
    thaw_schedule = options["training"].get("thaw_schedule")

    # Training state
    best_val_acc_top1 = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        # Check thaw schedule for progressive unfreezing
        new_thaw_pct = check_thaw_schedule(epoch, options)
        if new_thaw_pct is not None and new_thaw_pct > current_thaw_pct:
            # 1. Thaw backbone blocks (toggle requires_grad)
            thawed = thaw_backbone_percentage(model, backbone_name, new_thaw_pct)
            print(f"\n[Epoch {epoch}] Thawed {new_thaw_pct:.0%} of backbone: {thawed}")

            # 2. Update backbone LR in-place (preserves momentum state)
            if thaw_schedule:
                base_lr = options['training']['learning_rate']
                backbone_lr_ratio = options['training'].get('backbone_lr_ratio', 0.1)
                for group in optimizer.param_groups:
                    if group.get('name') == 'backbone':
                        group['lr'] = base_lr * backbone_lr_ratio

            current_thaw_pct = new_thaw_pct
            total, trainable = count_parameters(model)
            print(f"  Trainable: {trainable:,} / {total:,}\n")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )

        val_metrics = evaluate(model, val_loader, criterion, device)

        # Optuna pruning check
        if trial is not None:
            trial.report(val_metrics["acc_top1"], epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        current_lr = optimizer.param_groups[0]["lr"]
        is_best = early_stopping.step(val_metrics["acc_top1"])

        if is_best:
            best_val_acc_top1 = val_metrics["acc_top1"]
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
        "best_epoch": best_epoch,
        "final_test_acc_top1": test_metrics["acc_top1"],
        "inference_time_ms": inference_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    print_final_results(results)

    # Save results summary
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate visualizations
    print("\n[Visualization] Generating training plots...")
    try:
        plot_loss_curves(experiment_name, save=True, show=False, results_dir=results_dir)
        plot_accuracy_curves(experiment_name, save=True, show=False, results_dir=results_dir)
        plot_learning_rate(experiment_name, save=True, show=False, results_dir=results_dir)
        plot_experiment_summary(experiment_name, save=True, show=False, results_dir=results_dir)

        # Confusion matrix on test set
        all_targets, all_predictions = evaluate_with_predictions(
            model, test_loader, device
        )
        # Get class names from ImageFolder dataset
        class_names = getattr(test_loader.dataset, 'classes', None)
        if class_names is None:
            num_classes = options["model"]["num_classes"]
            class_names = [f"Class {i}" for i in range(num_classes)]
        plot_confusion_matrix(
            all_targets, all_predictions, class_names,
            results_dir, save=True, show=False
        )

        print(f"[Visualization] All plots saved to {results_dir}")
    except Exception as e:
        print(f"[Warning] Visualization failed: {e}")

    return results
