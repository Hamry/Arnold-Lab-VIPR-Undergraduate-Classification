"""
Training Engine Module

Stateless training function that takes a config, runs an experiment,
and saves results. Supports mixed precision and early stopping.
"""

import gc
import os
import json
import time
import random
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SLURM/headless

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model_utils import (
    load_model,
    count_parameters,
    _get_classifier_attr,
    get_backbone_blocks,
    thaw_backbone_percentage,
    get_unfreeze_units,
    thaw_units,
    UnfreezeUnit,
)
from .visualization import (
    plot_loss_curves,
    plot_accuracy_curves,
    plot_learning_rate,
    plot_experiment_summary,
    plot_confusion_matrix,
)


# ---------------------------------------------------------------------------
# Loss Function Helpers
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Multi-class focal loss (softmax/cross-entropy variant).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        alpha: Optional class weights. None, a scalar, or a list/tensor of
               length num_classes.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C] logits, targets: [B] long
        log_p = F.log_softmax(inputs, dim=1)
        p = torch.exp(log_p)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * log_pt
        if self.alpha is not None:
            alpha = self.alpha
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=inputs.dtype, device=inputs.device)
            at = alpha.gather(0, targets)
            loss = at * loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


class FocalLossSigmoid(nn.Module):
    """Per-class focal loss for multi-label sigmoid outputs (BCE variant).

    Args:
        gamma: Focusing parameter.
        alpha: Balance factor between positive/negative examples (0-1).
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C] logits, targets: [B, C] float one-hot
        p = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()


def build_criterion(loss_fn, model_opts=None):
    """
    Construct a loss criterion from a string identifier.

    Args:
        loss_fn: One of 'cross_entropy', 'bce_sigmoid', 'focal_loss',
                 'focal_loss_sigmoid'
        model_opts: The model config dict (used to read focal_gamma / focal_alpha).

    Returns:
        nn.Module: The loss criterion
    """
    if loss_fn == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_fn == "bce_sigmoid":
        return nn.BCEWithLogitsLoss()
    elif loss_fn == "focal_loss":
        gamma = model_opts.get("focal_gamma", 2.0) if model_opts else 2.0
        alpha = model_opts.get("focal_alpha", None) if model_opts else None
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif loss_fn == "focal_loss_sigmoid":
        gamma = model_opts.get("focal_gamma", 2.0) if model_opts else 2.0
        alpha = model_opts.get("focal_alpha", 0.25) if model_opts else 0.25
        return FocalLossSigmoid(gamma=gamma, alpha=alpha)
    else:
        raise ValueError(
            f"Unknown loss_fn: {loss_fn!r}. Supported values: "
            "'cross_entropy', 'bce_sigmoid', 'focal_loss', 'focal_loss_sigmoid'."
        )


def prepare_targets_for_loss(targets, num_classes, loss_fn):
    """
    Convert integer class indices to the format required by the criterion.

    CrossEntropyLoss expects a LongTensor of class indices [B].
    BCEWithLogitsLoss expects a float one-hot tensor [B, num_classes].

    Args:
        targets: LongTensor of class indices [B]
        num_classes: Total number of classes
        loss_fn: Loss function identifier string

    Returns:
        Tensor in the correct dtype/shape for the criterion
    """
    if loss_fn in ("bce_sigmoid", "focal_loss_sigmoid"):
        return nn.functional.one_hot(targets, num_classes).float()
    return targets


def logits_to_probs(outputs, loss_fn):
    """
    Convert raw model logits to class probabilities.

    Args:
        outputs: Raw logits [B, num_classes]
        loss_fn: Loss function identifier string

    Returns:
        Tensor of probabilities [B, num_classes]
    """
    if loss_fn in ("bce_sigmoid", "focal_loss_sigmoid"):
        return torch.sigmoid(outputs)
    return torch.softmax(outputs, dim=1)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
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
        persistent_workers=False,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    print(
        f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Optimizer and Scheduler
# ---------------------------------------------------------------------------


def create_optimizer(model, options, backbone_name=None, dynamic_unfreeze_mode=False):
    """
    Create optimizer from config with optional discriminative learning rates.

    When backbone_name is provided and thaw_schedule is configured, creates
    named parameter groups for backbone vs classifier with different LRs.
    Includes ALL parameters (even frozen ones) to preserve momentum state
    when thawing layers mid-training.

    When dynamic_unfreeze_mode=True, only the classifier params are included
    at startup. Backbone units are added later via optimizer.add_param_group()
    as they are unfrozen, giving each unit a fresh optimizer state.

    Args:
        model: The model
        options: Config dict
        backbone_name: If provided, enables discriminative LR setup
        dynamic_unfreeze_mode: If True, start with classifier params only.

    Returns:
        torch.optim.Optimizer
    """
    train_opts = options["training"]
    lr = train_opts["learning_rate"]
    weight_decay = train_opts.get("weight_decay", 0.01)
    optimizer_name = train_opts.get("optimizer", "adamw").lower()
    thaw_schedule = train_opts.get("thaw_schedule")

    if dynamic_unfreeze_mode and backbone_name:
        # Classifier-only startup; backbone units added incrementally
        classifier_attr = _get_classifier_attr(backbone_name)
        classifier = getattr(model, classifier_attr)
        param_groups = [
            {'params': list(classifier.parameters()), 'lr': lr, 'name': 'classifier'},
        ]
    elif backbone_name and thaw_schedule:
        # Use discriminative LR if thaw_schedule is configured
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
# Per-Class Metric Computation
# ---------------------------------------------------------------------------


def compute_per_class_metrics(targets, predictions, num_classes):
    """
    Compute per-class and macro-averaged classification metrics without sklearn.

    Args:
        targets: list of ground-truth class indices
        predictions: list of predicted class indices
        num_classes: total number of classes

    Returns:
        tuple:
            per_class_acc  (list[float]) – per-class recall / accuracy
            per_class_f1   (list[float]) – per-class F1 score
            per_class_prec (list[float]) – per-class precision
            per_class_rec  (list[float]) – per-class recall (same as acc)
            macro_f1       (float)
            macro_precision(float)
            macro_recall   (float)
    """
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(targets, predictions):
        cm[t][p] += 1

    acc_list, f1_list, prec_list, rec_list = [], [], [], []
    for i in range(num_classes):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[r][i] for r in range(num_classes)) - tp
        acc  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc_list.append(acc)
        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)

    macro_f1   = sum(f1_list)   / num_classes
    macro_prec = sum(prec_list) / num_classes
    macro_rec  = sum(rec_list)  / num_classes
    return acc_list, f1_list, prec_list, rec_list, macro_f1, macro_prec, macro_rec


# ---------------------------------------------------------------------------
# Training and Evaluation Loops
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device,
                    loss_fn="cross_entropy"):
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
            targets_for_loss = prepare_targets_for_loss(targets, outputs.size(1), loss_fn)
            loss = criterion(outputs, targets_for_loss)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, loss_fn="cross_entropy"):
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

            with autocast("cuda"):
                outputs = model(inputs)
                targets_for_loss = prepare_targets_for_loss(targets, outputs.size(1), loss_fn)
                loss = criterion(outputs, targets_for_loss)

            top1 = compute_accuracy(outputs, targets)

            total_loss += loss.item()
            total_top1 += top1
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "acc_top1": total_top1 / num_batches,
    }


def evaluate_with_predictions(model, loader, device, loss_fn="cross_entropy"):
    """
    Evaluate model and collect all predictions for confusion matrix.

    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation
        device: Device to run on
        loss_fn: Loss function identifier (unused here, kept for signature consistency)

    Returns:
        tuple: (all_targets, all_predictions) as lists
    """
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast("cuda"):
                outputs = model(inputs)

            _, predicted = outputs.max(1)
            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

    return all_targets, all_predictions


def evaluate_full(model, loader, criterion, device, class_names, loss_fn="cross_entropy"):
    """
    Evaluate model with full per-class and macro metrics in a single pass.

    Collects all predictions, computes loss and accuracy as usual, then
    derives per-class accuracy, per-class F1, and macro F1/precision/recall
    via compute_per_class_metrics.

    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run on
        class_names: List of class name strings (from dataset.classes)
        loss_fn: Loss function identifier string

    Returns:
        dict with keys:
            loss           (float)
            acc_top1       (float) – overall accuracy
            per_class_acc  (dict)  – {class_name: float}
            per_class_f1   (dict)  – {class_name: float}
            macro_f1       (float)
            macro_precision(float)
            macro_recall   (float)
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast("cuda"):
                outputs = model(inputs)
                targets_for_loss = prepare_targets_for_loss(targets, outputs.size(1), loss_fn)
                loss = criterion(outputs, targets_for_loss)

            _, predicted = outputs.max(1)
            total_loss += loss.item()
            num_batches += 1
            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

    num_classes = len(class_names)
    overall_acc = sum(t == p for t, p in zip(all_targets, all_predictions)) / len(all_targets)

    acc_list, f1_list, _, _, macro_f1, macro_prec, macro_rec = compute_per_class_metrics(
        all_targets, all_predictions, num_classes
    )

    per_class_acc = {class_names[i].lower(): acc_list[i] for i in range(num_classes)}
    per_class_f1  = {class_names[i].lower(): f1_list[i]  for i in range(num_classes)}

    return {
        "loss":            total_loss / num_batches,
        "acc_top1":        overall_acc,
        "per_class_acc":   per_class_acc,
        "per_class_f1":    per_class_f1,
        "macro_f1":        macro_f1,
        "macro_precision": macro_prec,
        "macro_recall":    macro_rec,
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


def init_metrics_file(results_dir, class_names):
    """Initialize the metrics CSV file with headers.

    Columns: epoch, train_loss, val_loss, val_acc_top1, lr,
             val_f1_<class> * N, val_acc_<class> * N
    """
    metrics_path = results_dir / "metrics.csv"
    headers = ["epoch", "train_loss", "val_loss", "val_acc_top1", "lr"]
    headers += [f"val_f1_{c.lower()}"  for c in class_names]
    headers += [f"val_acc_{c.lower()}" for c in class_names]
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    return metrics_path


def append_metrics(metrics_path, epoch, train_loss, val_metrics, lr, class_names):
    """Append one row of metrics to the CSV file."""
    row = [
        epoch,
        f"{train_loss:.4f}",
        f"{val_metrics['loss']:.4f}",
        f"{val_metrics['acc_top1']:.4f}",
        f"{lr:.6f}",
    ]
    # Per-class F1 then per-class accuracy, in class_names order
    for c in class_names:
        row.append(f"{val_metrics['per_class_f1'][c.lower()]:.4f}")
    for c in class_names:
        row.append(f"{val_metrics['per_class_acc'][c.lower()]:.4f}")
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


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
# Dynamic Thaw Controller
# ---------------------------------------------------------------------------


class DynamicThawController:
    """
    Plateau-triggered progressive unfreezing controller.

    Monitors validation loss after each epoch. When the loss fails to
    improve for `unfreeze_patience` consecutive epochs, the next batch
    of UnfreezeUnits (ordered output→input) is returned for thawing.

    Each newly unfrozen batch receives a learning rate of:
        lr = base_lr * lr_decay_ratio ** depth_level

    where depth_level starts at 1 for the first unfrozen batch and
    increments with each successive trigger.

    Usage::

        controller = DynamicThawController(units, patience=5, size=1,
                                           lr_decay_ratio=0.1, base_lr=1e-4)
        for epoch in ...:
            result = controller.step(val_loss)
            if result is not None:
                units_batch, unit_lr = result
                thaw_units(model, units_batch)
                optimizer.add_param_group({'params': ..., 'lr': unit_lr})
    """

    _MIN_DELTA = 0.001  # minimum loss improvement to reset counter

    def __init__(self, units, unfreeze_patience, unfreeze_size, lr_decay_ratio, base_lr):
        """
        Args:
            units: List[UnfreezeUnit] ordered output→input (from get_unfreeze_units).
            unfreeze_patience: Epochs without val_loss improvement before triggering.
            unfreeze_size: Number of units to open per trigger.
            lr_decay_ratio: LR multiplier per depth level (e.g. 0.1).
            base_lr: Classifier head learning rate (depth-0 reference).
        """
        self._units: List[UnfreezeUnit] = list(units)
        self._patience = unfreeze_patience
        self._size = unfreeze_size
        self._decay = lr_decay_ratio
        self._base_lr = base_lr

        self._best_loss: Optional[float] = None
        self._counter = 0
        self._next_idx = 0
        self._depth_level = 1  # depth 0 = classifier head; backbone starts at 1
        self.all_unfrozen = False

    def step(self, val_loss: float) -> Optional[Tuple[List[UnfreezeUnit], float]]:
        """
        Update controller state with the latest validation loss.

        Args:
            val_loss: Current epoch validation loss.

        Returns:
            (units_batch, lr) if a plateau was detected and units remain,
            otherwise None.
        """
        # Track best loss
        if self._best_loss is None or val_loss < self._best_loss - self._MIN_DELTA:
            self._best_loss = val_loss
            self._counter = 0
            return None

        self._counter += 1

        if self._counter < self._patience:
            return None

        # Plateau detected — check if units remain
        if self._next_idx >= len(self._units):
            self.all_unfrozen = True
            return None

        # Grab the next batch of units
        batch = self._units[self._next_idx: self._next_idx + self._size]
        unit_lr = self._base_lr * (self._decay ** self._depth_level)

        self._next_idx += len(batch)
        self._depth_level += 1
        self._counter = 0  # reset counter after unfreeze

        if self._next_idx >= len(self._units):
            self.all_unfrozen = True

        return batch, unit_lr


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
    print(f"Test Accuracy:  {results['final_test_acc_top1']:.2%}")
    print(f"Test F1 (macro):        {results['final_test_f1']:.4f}")
    print(f"Test Precision (macro): {results['final_test_precision']:.4f}")
    print(f"Test Recall (macro):    {results['final_test_recall']:.4f}")
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

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(options)
    class_names = getattr(train_loader.dataset, "classes", None)
    if class_names is None:
        num_classes = options["model"]["num_classes"]
        class_names = [f"class{i}" for i in range(num_classes)]

    metrics_path = init_metrics_file(results_dir, class_names)

    # Model
    model = load_model(options)
    model = model.to(device)
    backbone_name = options['model']['backbone']

    # Training components
    dynamic_unfreeze_opts = options["training"].get("dynamic_unfreeze")
    loss_fn = options["model"].get("loss_fn", "cross_entropy")
    criterion = build_criterion(loss_fn, model_opts=options["model"])
    optimizer = create_optimizer(
        model, options, backbone_name,
        dynamic_unfreeze_mode=bool(dynamic_unfreeze_opts)
    )
    scheduler = create_scheduler(optimizer, options, len(train_loader))
    scaler = GradScaler("cuda")

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Dynamic unfreezing controller (plateau-triggered)
    thaw_controller = None
    if dynamic_unfreeze_opts:
        unfreeze_units = get_unfreeze_units(model, backbone_name)
        print(f"[Dynamic Unfreeze] {len(unfreeze_units)} backbone block(s) available.")
        thaw_controller = DynamicThawController(
            units=unfreeze_units,
            unfreeze_patience=dynamic_unfreeze_opts["unfreeze_patience"],
            unfreeze_size=dynamic_unfreeze_opts["unfreeze_size"],
            lr_decay_ratio=dynamic_unfreeze_opts["lr_decay_ratio"],
            base_lr=options["training"]["learning_rate"],
        )

    # Progressive unfreezing state (epoch-schedule mode)
    current_thaw_pct = 0.0
    thaw_schedule = options["training"].get("thaw_schedule")

    # Training state
    best_val_acc_top1 = 0.0
    best_epoch = 0
    unfreeze_event_count = 0  # Tracks dynamic unfreeze events; used as Optuna pruning step

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
            model, train_loader, criterion, optimizer, scheduler, scaler, device,
            loss_fn=loss_fn
        )

        val_metrics = evaluate_full(model, val_loader, criterion, device, class_names,
                                    loss_fn=loss_fn)

        # Dynamic plateau-triggered unfreezing
        if thaw_controller is not None:
            result = thaw_controller.step(val_metrics["loss"])
            if result is not None:
                # Optuna pruning checkpoint: evaluate right before unfreezing.
                # Step = unfreeze event index so comparisons across trials are apples-to-apples.
                # Negate val_loss because the study direction is "maximize".
                if trial is not None:
                    import optuna
                    trial.report(-val_metrics["loss"], unfreeze_event_count)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                units_batch, unit_lr = result
                thaw_units(model, units_batch)
                new_params = [
                    p
                    for unit in units_batch
                    for name in unit.module_names
                    for p in model.get_submodule(name).parameters()
                ]
                depth = thaw_controller._depth_level - 1  # already incremented in step()
                optimizer.add_param_group({
                    'params': new_params,
                    'lr': unit_lr,
                    'name': f'backbone_d{depth}',
                })
                total, trainable = count_parameters(model)
                print(
                    f"\n[Epoch {epoch}] Dynamic unfreeze: {len(units_batch)} unit(s) "
                    f"@ lr={unit_lr:.2e} | Trainable: {trainable:,}/{total:,}\n"
                )
                unfreeze_event_count += 1

        # Epoch-based Optuna pruning (non-dynamic mode only).
        # Dynamic mode uses unfreeze-event pruning above.
        if trial is not None and thaw_controller is None:
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

        append_metrics(metrics_path, epoch, train_loss, val_metrics, current_lr, class_names)
        print_epoch_summary(epoch, epochs, train_loss, val_metrics, current_lr, is_best)

        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Load best model and evaluate on test set
    load_checkpoint(model, results_dir)
    test_metrics = evaluate_full(model, test_loader, criterion, device, class_names,
                                 loss_fn=loss_fn)

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
        "final_test_f1": test_metrics["macro_f1"],
        "final_test_precision": test_metrics["macro_precision"],
        "final_test_recall": test_metrics["macro_recall"],
        "inference_time_ms": inference_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "per_class_test_f1": test_metrics["per_class_f1"],
        "per_class_test_acc": test_metrics["per_class_acc"],
    }

    print_final_results(results)

    # Save results summary
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate visualizations (must happen before deleting DataLoaders)
    print("\n[Visualization] Generating training plots...")
    try:
        plot_loss_curves(experiment_name, save=True, show=False, results_dir=results_dir)
        plot_accuracy_curves(experiment_name, save=True, show=False, results_dir=results_dir)
        plot_learning_rate(experiment_name, save=True, show=False, results_dir=results_dir)
        plot_experiment_summary(experiment_name, save=True, show=False, results_dir=results_dir)

        # Confusion matrix on test set
        all_targets, all_predictions = evaluate_with_predictions(
            model, test_loader, device, loss_fn=loss_fn
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

    # Explicitly shut down DataLoader workers before returning.
    # With persistent_workers=True and pin_memory=True, workers and the
    # pin_memory_thread can deadlock during GC at trial boundaries unless
    # the loaders are deleted here while the CUDA context is still alive.
    del train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()

    return results
