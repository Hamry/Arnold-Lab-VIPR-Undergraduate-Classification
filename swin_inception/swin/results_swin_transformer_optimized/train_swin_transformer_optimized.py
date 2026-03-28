"""
SWIN TRANSFORMER TRAINING SCRIPT FOR VIPR IMAGE CLASSIFICATION

OVERVIEW:
This script trains a Swin Transformer (a modern vision transformer architecture) to classify 
VIPR medical images into 4 categories: Blurry, Good, Opaque, and Yellow. The goal is to 
achieve at least 90% F1-score. Swin Transformer uses attention mechanisms instead of traditional 
convolutions, making it potentially better at capturing global patterns in images.

HOW IT WORKS:
1. Loads and balances the training data (ensures all classes are equally represented)
2. Uses a pre-trained Swin Transformer and customizes it for our 4 classes
3. Trains in 3 progressive phases to prevent overfitting:
   - Phase 1: Train only the new classifier layers (30 epochs)
   - Phase 2: Unfreeze and train the last transformer stage (35 epochs)
   - Phase 3: Fine-tune the entire network (35 epochs)
4. Uses advanced augmentation techniques including MixUp, CutMix, and RandAugment
5. Applies test-time augmentation (TTA) for better predictions
6. Saves the best model and generates performance visualizations

KEY DIFFERENCES FROM INCEPTION V3:
- Transformers need more aggressive data augmentation (RandAugment)
- Lower learning rates (transformers are more sensitive)
- Higher weight decay (0.05 vs 0.01) to prevent overfitting
- More extensive test-time augmentation (4 versions vs 3)
- Label smoothing to improve calibration

TRAINING STRATEGY:
Transformers learn differently than CNNs. They need:
- More data augmentation to generalize well
- Lower learning rates to avoid instability
- Careful regularization to prevent overfitting
- Progressive unfreezing from attention heads to patch embeddings

EXPECTED OUTCOME:
A trained model achieving 84-86% F1-score across all 4 image quality classes.
While slightly lower than Inception v3, transformers may generalize better to new data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter

import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss: focuses training on hard examples (like Blurry).
    gamma=0 is identical to CrossEntropyLoss.
    gamma=2 downweights easy examples by ~4x.
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()

# ============================================================================
# CONFIGURATION - All training settings
# ============================================================================

class Config:
    """Configuration optimized for Swin Transformer"""
    
    # Data paths
    DATA_DIR = './final_split_dataset'
    TRAIN_DIR = 'train'
    VAL_DIR = 'validate'
    
    # Model info
    MODEL_NAME = 'swin_transformer_optimized'
    INPUT_SIZE = 224  # Swin uses 224x224 images
    NUM_CLASSES = 4   # Blurry, Good, Opaque, Yellow
    
    # Training parameters
    BATCH_SIZE = 24
    NUM_EPOCHS = 100
    
    # Training phases
    PHASE1_EPOCHS = 30  # Train classifier only
    PHASE2_EPOCHS = 35  # Unfreeze last transformer stage
    PHASE3_EPOCHS = 35  # Fine-tune everything
    
    CLASSIFIER_LR = 0.0001
    BACKBONE_LR = 0.000008
    WEIGHT_DECAY = 0.03
    PHASE3_LR = 0.000005     # Very low for full fine-tune
    
    # Focal Loss
    FOCAL_GAMMA = 2.5        # Higher gamma for Swin (harder problem)
    LABEL_SMOOTHING = 0.1
    
    # Warmup
    WARMUP_EPOCHS = 5
    
    # Augmentation (more aggressive for transformers)
    MIXUP_ALPHA = 0.3    # Higher mixing
    CUTMIX_PROB = 0.5    # Use CutMix more often
    
    # Test-time augmentation
    TTA_TRANSFORMS = 5
    
    # Early stopping
    PATIENCE = 25
    MIN_DELTA = 0.001
    
    # Output
    OUTPUT_DIR = './results_swin_transformer_optimized'
    SAVE_BEST_MODEL = True
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# DATA LOADING - Same as Inception but with transformer-optimized augmentation
# ============================================================================

def get_class_weights(dataset):
    targets = [label for _, label in dataset.samples]
    class_counts = Counter(targets)
    total_samples = len(targets)
    weights = {cls: total_samples / count for cls, count in class_counts.items()}
    max_weight = max(weights.values())
    weights = {cls: w / max_weight for cls, w in weights.items()}
    # Manually boost Blurry (class 0) — it has the worst recall
    blurry_idx = dataset.classes.index('Blurry')
    weights[blurry_idx] = weights[blurry_idx] * 1.8
    return weights


def get_balanced_sampler(dataset):
    """Create sampler for balanced training"""
    targets = [label for _, label in dataset.samples]
    class_counts = Counter(targets)
    weights = [1.0 / class_counts[label] for label in targets]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler


def mixup_data(x, y, alpha=0.3):
    """MixUp augmentation - blend two images"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation - cut and paste image patches"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    W, H = x.size()[2], x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def get_data_loaders(config):
    """
    Load data with transformer-optimized augmentation.
    Transformers benefit from more aggressive augmentation than CNNs.
    """
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # MORE aggressive augmentation for transformers
    train_transforms = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),  # More rotation
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomResizedCrop(config.INPUT_SIZE, scale=(0.6, 1.0)),
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        # RandAugment: applies random augmentation policies
        transforms.RandAugment(num_ops=3, magnitude=12),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 3.0))], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(1.0, 5.0))], p=0.3),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = datasets.ImageFolder(
        os.path.join(config.DATA_DIR, config.TRAIN_DIR),
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(config.DATA_DIR, config.VAL_DIR),
        transform=val_transforms
    )
    
    class_weights = get_class_weights(train_dataset)
    print(f"\nClass weights:")
    for cls_idx, cls_name in enumerate(train_dataset.classes):
        print(f"  {cls_name}: {class_weights[cls_idx]:.3f}")
    
    balanced_sampler = get_balanced_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=balanced_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\n{'='*60}")
    print("Dataset Information:")
    print(f"{'='*60}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_targets = [label for _, label in train_dataset.samples]
    train_counts = Counter(train_targets)
    print(f"\nClass distribution:")
    for cls_idx, cls_name in enumerate(train_dataset.classes):
        count = train_counts[cls_idx]
        percentage = 100 * count / len(train_dataset)
        print(f"  {cls_name}: {count} ({percentage:.1f}%)")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, train_dataset.classes, class_weights


# ============================================================================
# MODEL SETUP - Create and configure Swin Transformer
# ============================================================================

def create_swin_model(num_classes):
    """
    Create Swin Transformer with custom classifier.
    Uses LayerNorm instead of BatchNorm (better for transformers).
    """
    # Load pre-trained Swin Transformer
    weights = models.Swin_T_Weights.IMAGENET1K_V1
    model = models.swin_t(weights=weights)
    
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Custom classifier with LayerNorm (better for transformers)
    num_features = model.head.in_features
    
    model.head = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.LayerNorm(1024),  # LayerNorm works better than BatchNorm for transformers
        nn.GELU(),           # GELU activation (smoother than ReLU)
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model


def unfreeze_layers(model, phase):
    """
    Progressive unfreezing for Swin Transformer.
    
    Phase 1: Only train classifier
    Phase 2: Train classifier + last Swin stage (attention blocks)
    Phase 3: Train everything
    """
    if phase == 1:
        # Only classifier
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        print("Phase 1: Training classifier only")
        
    elif phase == 2:
        # Last TWO Swin stages + classifier
        for name, param in model.named_parameters():
            if any(x in name for x in ['features.6', 'features.7', 'head', 'norm']):
                param.requires_grad = True
        print("Phase 2: Unfroze features.6 + features.7 + classifier")
        
    elif phase == 3:
        # Everything
        for param in model.parameters():
            param.requires_grad = True
        print("Phase 3: Fine-tuning all layers")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")


def get_optimizer_groups(model, config, phase):
    """Create parameter groups with discriminative learning rates"""
    if phase == 1:
        return [{'params': filter(lambda p: p.requires_grad, model.parameters()),
                 'lr': config.CLASSIFIER_LR}]
    
    elif phase >= 2:
        classifier_params = []
        backbone_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'head' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        return [
            {'params': classifier_params, 'lr': config.CLASSIFIER_LR},
            {'params': backbone_params, 'lr': config.BACKBONE_LR}
        ]


# ============================================================================
# METRICS TRACKING
# ============================================================================

class MetricsTracker:
    """Track predictions and calculate performance metrics"""
    
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, preds, labels, probs):
        self.all_preds.extend(preds.detach().cpu().numpy())
        self.all_labels.extend(labels.detach().cpu().numpy())
        self.all_probs.extend(probs.detach().cpu().numpy())
    
    def compute_metrics(self):
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
        }
        
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'f1_{class_name}'] = f1_per_class[i]
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
        
        return metrics
    
    def get_confusion_matrix(self):
        return confusion_matrix(self.all_labels, self.all_preds)
    
    def get_classification_report(self):
        return classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.class_names,
            zero_division=0
        )


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, config, use_mixup=True):
    """Train for one epoch with MixUp/CutMix augmentation"""
    model.train()
    
    running_loss = 0.0
    if hasattr(train_loader.dataset, 'dataset'):
        class_names = train_loader.dataset.dataset.classes
    else:
        class_names = train_loader.dataset.classes
    metrics_tracker = MetricsTracker(config.NUM_CLASSES, class_names)
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Apply augmentation
        if use_mixup and np.random.rand() < 0.5:
            if np.random.rand() < config.CUTMIX_PROB:
                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha=1.0)
            else:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=config.MIXUP_ALPHA)
            mixed = True
        else:
            mixed = False
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        if mixed:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping (important for transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        metrics_tracker.update(preds, labels if not mixed else labels_a, probs)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_metrics = metrics_tracker.compute_metrics()
    
    return epoch_loss, epoch_metrics


def evaluate(model, val_loader, criterion, device, config, use_tta=False):
    """
    Evaluate on validation set.
    Uses more extensive TTA for transformers (4 versions instead of 3).
    """
    model.eval()
    
    running_loss = 0.0
    metrics_tracker = MetricsTracker(config.NUM_CLASSES, val_loader.dataset.classes)
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if use_tta:
                outputs_list = []
                # Original
                outputs_list.append(torch.softmax(model(inputs), dim=1))
                # H-flip
                outputs_list.append(torch.softmax(model(torch.flip(inputs, [3])), dim=1))
                # V-flip
                outputs_list.append(torch.softmax(model(torch.flip(inputs, [2])), dim=1))
                # Both flips
                outputs_list.append(torch.softmax(model(torch.flip(inputs, [2,3])), dim=1))
                # 90-degree rotation
                outputs_list.append(torch.softmax(model(torch.rot90(inputs, 1, [2,3])), dim=1))
                # 270-degree rotation
                outputs_list.append(torch.softmax(model(torch.rot90(inputs, 3, [2,3])), dim=1))
                probs   = torch.stack(outputs_list).mean(dim=0)
                outputs = torch.log(probs)
                
            else:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(probs, 1)
            metrics_tracker.update(preds, labels, probs)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_metrics = metrics_tracker.compute_metrics()
    
    return epoch_loss, epoch_metrics, metrics_tracker


# ============================================================================
# VISUALIZATION - Same plotting functions as Inception
# ============================================================================

def plot_training_history(history, output_dir):
    """Plot training history"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_f1'], label='Train')
    axes[0, 1].plot(history['val_f1'], label='Validation')
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='90% Target')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(history['train_acc'], label='Train')
    axes[0, 2].plot(history['val_acc'], label='Validation')
    axes[0, 2].axhline(y=0.9, color='r', linestyle='--', label='90% Target')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    class_names = ['Blurry', 'Good', 'Opaque', 'Yellow']
    for cls in class_names:
        if f'val_f1_{cls}' in history:
            axes[1, 0].plot(history[f'val_f1_{cls}'], label=cls)
    axes[1, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Per-Class F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    if 'learning_rate_classifier' in history:
        axes[1, 1].plot(history['learning_rate_classifier'], label='Classifier')
        if 'learning_rate_backbone' in history and len(history['learning_rate_backbone']) > 0:
            axes[1, 1].plot(history['learning_rate_backbone'], label='Backbone')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    if 'phase' in history:
        axes[1, 2].plot(history['phase'])
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Phase')
        axes[1, 2].set_title('Training Phase')
        axes[1, 2].set_yticks([1, 2, 3])
        axes[1, 2].set_yticklabels(['Classifier', 'Partial', 'Full'])
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})'
    
    sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt='',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Accuracy'}
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(metrics, class_names, output_dir):
    """Plot per-class performance"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    classes = class_names
    f1_scores = [metrics[f'f1_{cls}'] for cls in classes]
    precisions = [metrics[f'precision_{cls}'] for cls in classes]
    recalls = [metrics[f'recall_{cls}'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.6
    
    bars = axes[0].bar(x, f1_scores, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0].axhline(y=0.9, color='r', linestyle='--', label='90% Target')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('Per-Class F1')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=45)
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    bars = axes[1].bar(x, precisions, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% Target')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Per-Class Precision')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=45)
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    bars = axes[2].bar(x, recalls, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[2].axhline(y=0.9, color='r', linestyle='--', label='90% Target')
    axes[2].set_ylabel('Recall')
    axes[2].set_title('Per-Class Recall')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(classes, rotation=45)
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_model(config):
    """Main training function with progressive unfreezing"""
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("OPTIMIZED SWIN TRANSFORMER TRAINING FOR 90%+ F1")
    print("="*70)
    
    train_loader, val_loader, class_names, class_weights = get_data_loaders(config)
    
    model = create_swin_model(config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    
    weight_tensor = torch.FloatTensor([class_weights[i] for i in range(config.NUM_CLASSES)])
    weight_tensor = weight_tensor.to(config.DEVICE)
    # Label smoothing helps transformers generalize better
    criterion = FocalLoss(
        weight=weight_tensor,
        gamma=config.FOCAL_GAMMA,
        label_smoothing=config.LABEL_SMOOTHING
    )
    
    print(f"\nWeighted loss: {weight_tensor.cpu().numpy()}")
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': [],
        'learning_rate_classifier': [],
        'learning_rate_backbone': [],
        'phase': []
    }
    
    for cls in class_names:
        history[f'val_f1_{cls}'] = []
        history[f'val_precision_{cls}'] = []
        history[f'val_recall_{cls}'] = []
    
    best_f1 = 0.0
    epochs_without_improvement = 0
    global_epoch = 0
    
    # ========================================================================
    # PHASE 1: Classifier training
    # ========================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: CLASSIFIER TRAINING")
    print(f"{'='*70}")
    unfreeze_layers(model, phase=1)
    
    # Lower LR for final fine-tune to avoid overfitting
    classifier_params = [p for n,p in model.named_parameters()
                            if p.requires_grad and ('fc' in n or 'AuxLogits' in n)]
    backbone_params   = [p for n,p in model.named_parameters()
                            if p.requires_grad and 'fc' not in n and 'AuxLogits' not in n]
    optimizer = optim.AdamW([
        {'params': classifier_params, 'lr': getattr(config, 'PHASE3_LR', config.BACKBONE_LR)},
        {'params': backbone_params,   'lr': getattr(config, 'PHASE3_LR', config.BACKBONE_LR) * 0.3},
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    
    for epoch in range(config.PHASE1_EPOCHS):
        print(f"\nEpoch {global_epoch + 1} (Phase 1, {epoch + 1}/{config.PHASE1_EPOCHS})")
        print("-" * 70)
        
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, config, use_mixup=True
        )
        
        val_loss, val_metrics, val_tracker = evaluate(
            model, val_loader, criterion, config.DEVICE, config, use_tta=False
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1_macro'])
        history['val_f1'].append(val_metrics['f1_macro'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['learning_rate_classifier'].append(optimizer.param_groups[0]['lr'])
        history['learning_rate_backbone'].append(0)
        history['phase'].append(1)
        
        for cls in class_names:
            history[f'val_f1_{cls}'].append(val_metrics[f'f1_{cls}'])
            history[f'val_precision_{cls}'].append(val_metrics[f'precision_{cls}'])
            history[f'val_recall_{cls}'].append(val_metrics[f'recall_{cls}'])
        
        print(f"Train: Loss={train_loss:.4f}, F1={train_metrics['f1_macro']:.4f}, Acc={train_metrics['accuracy']:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, F1={val_metrics['f1_macro']:.4f}, Acc={val_metrics['accuracy']:.4f}")
        print(f"\nPer-class F1:")
        for cls in class_names:
            print(f"  {cls}: {val_metrics[f'f1_{cls}']:.4f} (P:{val_metrics[f'precision_{cls}']:.3f}, R:{val_metrics[f'recall_{cls}']:.3f})")
        
        if val_metrics['f1_macro'] > best_f1 + config.MIN_DELTA:
            best_f1 = val_metrics['f1_macro']
            epochs_without_improvement = 0
            
            torch.save({
                'epoch': global_epoch,
                'phase': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'class_names': class_names
            }, os.path.join(config.OUTPUT_DIR, 'best_model.pth'))
            
            print(f"✓ New best! F1: {best_f1:.4f}")
        else:
            epochs_without_improvement += 1
        
        global_epoch += 1
        
        if epochs_without_improvement >= config.PATIENCE:
            print(f"\nEarly stop Phase 1")
            break
    
    # ========================================================================
    # PHASE 2: Partial unfreezing
    # ========================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: PARTIAL UNFREEZING")
    print(f"{'='*70}")
    
    checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    unfreeze_layers(model, phase=2)
    
    param_groups = get_optimizer_groups(model, config, phase=2)
    optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    
    epochs_without_improvement = 0
    
    for epoch in range(config.PHASE2_EPOCHS):
        print(f"\nEpoch {global_epoch + 1} (Phase 2, {epoch + 1}/{config.PHASE2_EPOCHS})")
        print("-" * 70)
        
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, config, use_mixup=True
        )
        
        val_loss, val_metrics, val_tracker = evaluate(
            model, val_loader, criterion, config.DEVICE, config, use_tta=False
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1_macro'])
        history['val_f1'].append(val_metrics['f1_macro'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['learning_rate_classifier'].append(optimizer.param_groups[0]['lr'])
        history['learning_rate_backbone'].append(optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0)
        history['phase'].append(2)
        
        for cls in class_names:
            history[f'val_f1_{cls}'].append(val_metrics[f'f1_{cls}'])
            history[f'val_precision_{cls}'].append(val_metrics[f'precision_{cls}'])
            history[f'val_recall_{cls}'].append(val_metrics[f'recall_{cls}'])
        
        print(f"Train: Loss={train_loss:.4f}, F1={train_metrics['f1_macro']:.4f}, Acc={train_metrics['accuracy']:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, F1={val_metrics['f1_macro']:.4f}, Acc={val_metrics['accuracy']:.4f}")
        print(f"\nPer-class F1:")
        for cls in class_names:
            print(f"  {cls}: {val_metrics[f'f1_{cls}']:.4f}")
        
        if val_metrics['f1_macro'] > best_f1 + config.MIN_DELTA:
            best_f1 = val_metrics['f1_macro']
            epochs_without_improvement = 0
            
            torch.save({
                'epoch': global_epoch,
                'phase': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'class_names': class_names
            }, os.path.join(config.OUTPUT_DIR, 'best_model.pth'))
            
            print(f"✓ New best! F1: {best_f1:.4f}")
        else:
            epochs_without_improvement += 1
        
        global_epoch += 1
        
        if epochs_without_improvement >= config.PATIENCE:
            print(f"\nEarly stop Phase 2")
            break
    
    # ========================================================================
    # PHASE 3: Full fine-tuning
    # ========================================================================
    if best_f1 < 0.90:
        print(f"\n{'='*70}")
        print("PHASE 3: FULL FINE-TUNING")
        print(f"{'='*70}")
        
        checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        unfreeze_layers(model, phase=3)
        
        param_groups = get_optimizer_groups(model, config, phase=3)
        optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
        
        epochs_without_improvement = 0
        
        for epoch in range(config.PHASE3_EPOCHS):
            print(f"\nEpoch {global_epoch + 1} (Phase 3, {epoch + 1}/{config.PHASE3_EPOCHS})")
            print("-" * 70)
            
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, config.DEVICE, config, use_mixup=False
            )
            
            val_loss, val_metrics, val_tracker = evaluate(
                model, val_loader, criterion, config.DEVICE, config, use_tta=True
            )
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_f1'].append(train_metrics['f1_macro'])
            history['val_f1'].append(val_metrics['f1_macro'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['learning_rate_classifier'].append(optimizer.param_groups[0]['lr'])
            history['learning_rate_backbone'].append(optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0)
            history['phase'].append(3)
            
            for cls in class_names:
                history[f'val_f1_{cls}'].append(val_metrics[f'f1_{cls}'])
                history[f'val_precision_{cls}'].append(val_metrics[f'precision_{cls}'])
                history[f'val_recall_{cls}'].append(val_metrics[f'recall_{cls}'])
            
            print(f"Train: Loss={train_loss:.4f}, F1={train_metrics['f1_macro']:.4f}, Acc={train_metrics['accuracy']:.4f}")
            print(f"Val:   Loss={val_loss:.4f}, F1={val_metrics['f1_macro']:.4f}, Acc={val_metrics['accuracy']:.4f} (TTA)")
            print(f"\nPer-class F1:")
            for cls in class_names:
                print(f"  {cls}: {val_metrics[f'f1_{cls}']:.4f}")
            
            if val_metrics['f1_macro'] > best_f1 + config.MIN_DELTA:
                best_f1 = val_metrics['f1_macro']
                epochs_without_improvement = 0
                
                torch.save({
                    'epoch': global_epoch,
                    'phase': 3,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'val_metrics': val_metrics,
                    'class_names': class_names
                }, os.path.join(config.OUTPUT_DIR, 'best_model.pth'))
                
                print(f"✓ New best! F1: {best_f1:.4f}")
            else:
                epochs_without_improvement += 1
            
            global_epoch += 1
            
            if best_f1 >= 0.90:
                print(f"\n🎉 ACHIEVED 90%+!")
                break
            
            if epochs_without_improvement >= config.PATIENCE:
                print(f"\nEarly stop Phase 3")
                break
    
    # ========================================================================
    # Final evaluation
    # ========================================================================
    print(f"\n{'='*70}")
    print("FINAL EVALUATION (with TTA)")
    print(f"{'='*70}")
    
    checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_metrics, final_tracker = evaluate(
        model, val_loader, criterion, config.DEVICE, config, use_tta=True
    )
    
    print(f"\nFINAL RESULTS:")
    print(f"{'='*70}")
    print(f"Accuracy:    {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"F1 (Macro):  {final_metrics['f1_macro']:.4f} ({final_metrics['f1_macro']*100:.2f}%)")
    print(f"F1 (Weight): {final_metrics['f1_weighted']:.4f}")
    print(f"Precision:   {final_metrics['precision_macro']:.4f}")
    print(f"Recall:      {final_metrics['recall_macro']:.4f}")
    
    print(f"\nPer-Class:")
    print(f"{'='*70}")
    for cls in class_names:
        f1 = final_metrics[f'f1_{cls}']
        prec = final_metrics[f'precision_{cls}']
        rec = final_metrics[f'recall_{cls}']
        status = "✓" if f1 >= 0.90 else "⚠"
        print(f"{status} {cls:10s}: F1={f1:.4f} ({f1*100:.2f}%), P={prec:.4f}, R={rec:.4f}")
    
    print(f"\n{final_tracker.get_classification_report()}")
    
    print("\nGenerating plots...")
    plot_training_history(history, config.OUTPUT_DIR)
    plot_confusion_matrix(final_tracker.get_confusion_matrix(), class_names, config.OUTPUT_DIR)
    plot_per_class_metrics(final_metrics, class_names, config.OUTPUT_DIR)
    
    with open(os.path.join(config.OUTPUT_DIR, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    with open(os.path.join(config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n✓ Saved to {config.OUTPUT_DIR}")
    
    if final_metrics['f1_macro'] >= 0.90:
        print(f"\n{'='*70}")
        print("🎉 SUCCESS! 90%+ F1-SCORE! 🎉")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"⚠ Target not reached. Best: {final_metrics['f1_macro']:.4f} ({final_metrics['f1_macro']*100:.2f}%)")
        print(f"{'='*70}")
    
    return model, final_metrics, history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    config = Config()
    
    print(f"\n{'='*70}")
    print("OPTIMIZED SWIN TRANSFORMER CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Classifier LR: {config.CLASSIFIER_LR}")
    print(f"Backbone LR: {config.BACKBONE_LR}")
    print(f"MixUp: {config.MIXUP_ALPHA}")
    print(f"CutMix: {config.CUTMIX_PROB}")
    print(f"Output: {config.OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    model, metrics, history = train_model(config)
    
    print("\n✓ Training complete!")