"""
INCEPTION V3 TRAINING SCRIPT FOR VIPR IMAGE CLASSIFICATION

OVERVIEW:
This script trains an Inception v3 deep learning model to classify VIPR medical images 
into 4 categories: Blurry, Good, Opaque, and Yellow. The goal is to achieve at least 
90% F1-score (a measure of accuracy that balances precision and recall).

HOW IT WORKS:
1. Loads and balances the training data (so all classes are equally represented)
2. Uses a pre-trained Inception v3 model and customizes it for our 4 classes
3. Trains in 3 progressive phases to prevent overfitting:
   - Phase 1: Train only the new classifier layers (30 epochs)
   - Phase 2: Unfreeze and train the last Inception block (30 epochs)
   - Phase 3: Fine-tune the entire network (40 epochs)
4. Uses advanced techniques like MixUp and CutMix to improve generalization
5. Applies test-time augmentation (TTA) for better predictions
6. Saves the best model and generates performance visualizations

KEY FEATURES:
- Class-balanced loss: Gives more weight to underrepresented classes
- Progressive unfreezing: Prevents catastrophic forgetting of pre-trained features
- MixUp/CutMix: Mixes training images to create harder examples
- Discriminative learning rates: Uses different learning rates for different layers
- Test-time augmentation: Makes predictions on multiple versions of each image
- Early stopping: Stops training if performance doesn't improve for 20 epochs

TRAINING STRATEGY:
The model learns gradually - first the classifier, then deeper layers, then everything.
This prevents overfitting and helps the model generalize better to new images.

EXPECTED OUTCOME:
A trained model achieving 89-90% F1-score across all 4 image quality classes.
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
    confusion_matrix, classification_report, roc_curve, auc
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
# CONFIGURATION - All training settings in one place
# ============================================================================

class Config:
    """All configuration settings for the training process"""
    
    # Where to find the data
    DATA_DIR = './final_split_dataset'
    TRAIN_DIR = 'train'
    VAL_DIR = 'validate'
    
    # Model info
    MODEL_NAME = 'inception_v3_optimized'
    INPUT_SIZE = 299  # Inception v3 requires 299x299 images
    NUM_CLASSES = 4   # Blurry, Good, Opaque, Yellow
    
    # How many images to process at once
    BATCH_SIZE = 24
    NUM_EPOCHS = 100  # Maximum epochs (early stopping may end sooner)
    
    # Training phases (progressive unfreezing)
    PHASE1_EPOCHS = 30  # Train classifier only
    PHASE2_EPOCHS = 30  # Unfreeze last Inception block
    PHASE3_EPOCHS = 40  # Fine-tune everything
    

    CLASSIFIER_LR = 0.0005  # Slightly lower for stability
    BACKBONE_LR = 0.00003   # Lower backbone LR
    WEIGHT_DECAY = 0.005    # Slightly reduced

    PHASE3_LR = 0.00008     # Very low for final fine-tune
    
    # Data augmentation settings
    MIXUP_ALPHA = 0.2    # How much to mix training images
    CUTMIX_PROB = 0.3    # Probability of using CutMix
    
    # Focal Loss
    FOCAL_GAMMA = 2.0    # Focus on hard examples (Blurry)
    LABEL_SMOOTHING = 0.1
    
    # Warmup
    WARMUP_EPOCHS = 5


    # Test-time augmentation
    TTA_TRANSFORMS = 5   # Number of augmented versions per image
    
    # Early stopping (prevents wasting time)
    PATIENCE = 25        # Stop if no improvement for 20 epochs
    MIN_DELTA = 0.001    # Minimum improvement to count
    
    # Where to save results
    OUTPUT_DIR = './results_inception_v3_optimized'
    SAVE_BEST_MODEL = True
    
    # Use GPU if available, otherwise CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# DATA LOADING - Load and balance the training data
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
    """
    Create a sampler that gives equal importance to all classes.
    This prevents the model from ignoring minority classes.
    """
    targets = [label for _, label in dataset.samples]
    class_counts = Counter(targets)
    
    # Each sample's weight is inverse of its class frequency
    weights = [1.0 / class_counts[label] for label in targets]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    return sampler


def mixup_data(x, y, alpha=0.2):
    """
    MixUp augmentation: blend two images and their labels.
    This creates harder training examples and improves generalization.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # Random mixing coefficient
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)  # Random pairing

    # Mix images: new_image = lam * image1 + (1-lam) * image2
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation: cut a patch from one image and paste onto another.
    The label is mixed based on the area of the patch.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Calculate random box size and position
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

    # Replace the box with patch from another image
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match actual pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def get_data_loaders(config):
    """
    Load the training and validation data with augmentation.
    Returns data loaders that can be iterated over in batches.
    """
    
    # Standard normalization for ImageNet pre-trained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training augmentation: aggressive transformations to prevent overfitting
    train_transforms = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),      # Flip left-right
        transforms.RandomVerticalFlip(p=0.5),        # Flip up-down
        transforms.RandomRotation(25),               # Rotate up to 25 degrees
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(config.INPUT_SIZE, scale=(0.7, 1.0)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 3.0))], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(1.0, 5.0))], p=0.3),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Random patches erased
    ])
    
    # Validation: no augmentation, just resize and normalize
    val_transforms = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Load the datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(config.DATA_DIR, config.TRAIN_DIR),
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(config.DATA_DIR, config.VAL_DIR),
        transform=val_transforms
    )
    
    # Calculate class weights for balanced training
    class_weights = get_class_weights(train_dataset)
    print(f"\nClass weights for balanced loss:")
    for cls_idx, cls_name in enumerate(train_dataset.classes):
        print(f"  {cls_name}: {class_weights[cls_idx]:.3f}")
    
    # Create balanced sampler
    balanced_sampler = get_balanced_sampler(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=balanced_sampler,  # Use balanced sampling
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
    
    # Print dataset info
    print(f"\n{'='*60}")
    print("Dataset Information:")
    print(f"{'='*60}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Show class distribution
    train_targets = [label for _, label in train_dataset.samples]
    train_counts = Counter(train_targets)
    print(f"\nTraining class distribution:")
    for cls_idx, cls_name in enumerate(train_dataset.classes):
        count = train_counts[cls_idx]
        percentage = 100 * count / len(train_dataset)
        print(f"  {cls_name}: {count} ({percentage:.1f}%)")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, train_dataset.classes, class_weights


# ============================================================================
# MODEL SETUP - Create and configure the Inception v3 model
# ============================================================================

def create_inception_model(num_classes):
    """
    Create Inception v3 with a custom classifier for our task.
    Starts with ImageNet pre-trained weights.
    """
    # Load pre-trained Inception v3
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights)
    
    # Freeze all layers initially (will unfreeze later)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the classifier with our custom one
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),  # Normalizes activations
        nn.ReLU(),
        nn.Dropout(0.5),       # Prevents overfitting
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)  # Final output: 4 classes
    )
    
    # Also update the auxiliary classifier (Inception v3 specific)
    if hasattr(model, 'AuxLogits'):
        num_aux_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_aux_features, num_classes)
    
    return model


def unfreeze_layers(model, phase):
    """
    Progressive unfreezing: gradually train more layers as training progresses.
    
    Phase 1: Only train the new classifier
    Phase 2: Train classifier + last Inception block
    Phase 3: Train everything (full fine-tuning)
    """
    if phase == 1:
        # Freeze everything except classifier
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        if hasattr(model, 'AuxLogits'):
            for param in model.AuxLogits.parameters():
                param.requires_grad = True
        print("Phase 1: Training classifier only")
        
    elif phase == 2:
        # Unfreeze last TWO Inception blocks + classifier
        for name, param in model.named_parameters():
            if any(x in name for x in ['Mixed_6', 'Mixed_7', 'fc', 'AuxLogits']):
                param.requires_grad = True
        print("Phase 2: Unfroze Mixed_6 + Mixed_7 blocks + classifier")
        
    elif phase == 3:
        # Unfreeze everything for final fine-tuning
        for param in model.parameters():
            param.requires_grad = True
        print("Phase 3: Fine-tuning all layers")
    
    # Show how many parameters are trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")


def get_optimizer_groups(model, config, phase):
    """
    Create parameter groups with different learning rates.
    New layers (classifier) get higher LR, pre-trained layers get lower LR.
    """
    if phase == 1:
        # Phase 1: Only classifier is trainable
        return [{'params': filter(lambda p: p.requires_grad, model.parameters()),
                 'lr': config.CLASSIFIER_LR}]
    
    elif phase >= 2:
        # Phase 2+: Different LRs for classifier vs backbone
        classifier_params = []
        backbone_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'fc' in name or 'AuxLogits' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        return [
            {'params': classifier_params, 'lr': config.CLASSIFIER_LR},
            {'params': backbone_params, 'lr': config.BACKBONE_LR}
        ]


# ============================================================================
# METRICS TRACKING - Calculate and store performance metrics
# ============================================================================

class MetricsTracker:
    """Tracks predictions and calculates various performance metrics"""
    
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Clear all stored predictions"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, preds, labels, probs):
        """Store a batch of predictions"""
        self.all_preds.extend(preds.detach().cpu().numpy())
        self.all_labels.extend(labels.detach().cpu().numpy())
        self.all_probs.extend(probs.detach().cpu().numpy())
    
    def compute_metrics(self):
        """Calculate all performance metrics"""
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # Overall metrics
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'f1_{class_name}'] = f1_per_class[i]
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
        
        return metrics
    
    def get_confusion_matrix(self):
        """Return confusion matrix"""
        return confusion_matrix(self.all_labels, self.all_preds)
    
    def get_classification_report(self):
        """Return detailed classification report"""
        return classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.class_names,
            zero_division=0
        )


# ============================================================================
# TRAINING - Train the model for one epoch
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, config, use_mixup=True):
    """
    Train the model for one complete pass through the training data.
    Uses MixUp/CutMix augmentation if enabled.
    """
    model.train()  # Set to training mode (enables dropout, etc.)
    
    running_loss = 0.0
    if hasattr(train_loader.dataset, 'dataset'):
        class_names = train_loader.dataset.dataset.classes
    else:
        class_names = train_loader.dataset.classes
    metrics_tracker = MetricsTracker(config.NUM_CLASSES, class_names)
    
    for inputs, labels in train_loader:
        # Move data to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Apply MixUp or CutMix randomly
        if use_mixup and np.random.rand() < 0.5:
            if np.random.rand() < config.CUTMIX_PROB:
                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha=1.0)
            else:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=config.MIXUP_ALPHA)
            mixed = True
        else:
            mixed = False
        
        optimizer.zero_grad()  # Clear previous gradients
        
        # Forward pass
        outputs = model(inputs)
        
        # Inception v3 has auxiliary output during training
        if isinstance(outputs, tuple):
            main_output = outputs[0]
            aux_output = outputs[1]
            
            if mixed:
                # Mixed labels: loss = lam * loss(pred, label_a) + (1-lam) * loss(pred, label_b)
                loss1 = lam * criterion(main_output, labels_a) + (1 - lam) * criterion(main_output, labels_b)
                loss2 = lam * criterion(aux_output, labels_a) + (1 - lam) * criterion(aux_output, labels_b)
            else:
                loss1 = criterion(main_output, labels)
                loss2 = criterion(aux_output, labels)
            
            loss = loss1 + 0.4 * loss2  # Weighted combination
        else:
            if mixed:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            main_output = outputs
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()  # Update weights
        
        # Track statistics
        running_loss += loss.item() * inputs.size(0)
        
        # Get predictions for metrics
        probs = torch.softmax(main_output, dim=1)
        _, preds = torch.max(main_output, 1)
        metrics_tracker.update(preds, labels if not mixed else labels_a, probs)
    
    # Calculate average loss and metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_metrics = metrics_tracker.compute_metrics()
    
    return epoch_loss, epoch_metrics


def evaluate(model, val_loader, criterion, device, config, use_tta=False):
    """
    Evaluate the model on validation data.
    Can use test-time augmentation (TTA) for better accuracy.
    """
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    
    running_loss = 0.0
    metrics_tracker = MetricsTracker(config.NUM_CLASSES, val_loader.dataset.classes)
    
    with torch.no_grad():  # Don't calculate gradients (saves memory)
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
# VISUALIZATION - Create plots to visualize training progress
# ============================================================================

def plot_training_history(history, output_dir):
    """Create comprehensive plots of training history"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss over time
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1 Score over time
    axes[0, 1].plot(history['train_f1'], label='Train')
    axes[0, 1].plot(history['val_f1'], label='Validation')
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='90% Target')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy over time
    axes[0, 2].plot(history['train_acc'], label='Train')
    axes[0, 2].plot(history['val_acc'], label='Validation')
    axes[0, 2].axhline(y=0.9, color='r', linestyle='--', label='90% Target')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Accuracy Over Time')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Per-class F1 scores
    class_names = ['Blurry', 'Good', 'Opaque', 'Yellow']
    for cls in class_names:
        if f'val_f1_{cls}' in history:
            axes[1, 0].plot(history[f'val_f1_{cls}'], label=cls)
    axes[1, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Per-Class F1 Scores')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate schedule
    if 'learning_rate_classifier' in history:
        axes[1, 1].plot(history['learning_rate_classifier'], label='Classifier')
        if 'learning_rate_backbone' in history and len(history['learning_rate_backbone']) > 0:
            axes[1, 1].plot(history['learning_rate_backbone'], label='Backbone')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Training phase
    if 'phase' in history:
        axes[1, 2].plot(history['phase'])
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Training Phase')
        axes[1, 2].set_title('Progressive Unfreezing Phase')
        axes[1, 2].set_yticks([1, 2, 3])
        axes[1, 2].set_yticklabels(['Classifier Only', 'Partial Unfreeze', 'Full Fine-tune'])
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, output_dir):
    """Create a confusion matrix visualization"""
    plt.figure(figsize=(10, 8))
    
    # Normalize by row (actual class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotations with both count and percentage
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
        cbar_kws={'label': 'Normalized Accuracy'}
    )
    plt.title('Confusion Matrix (Normalized by True Class)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(metrics, class_names, output_dir):
    """Create bar charts showing per-class performance"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    classes = class_names
    f1_scores = [metrics[f'f1_{cls}'] for cls in classes]
    precisions = [metrics[f'precision_{cls}'] for cls in classes]
    recalls = [metrics[f'recall_{cls}'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.6
    
    # F1 Score
    bars = axes[0].bar(x, f1_scores, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0].axhline(y=0.9, color='r', linestyle='--', label='90% Target')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('Per-Class F1 Scores')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=45)
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    # Precision
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
    
    # Recall
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
# MAIN TRAINING LOOP - Orchestrates the entire training process
# ============================================================================

def train_model(config):
    """
    Main training function with progressive unfreezing.
    Trains in 3 phases with early stopping.
    """
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("OPTIMIZED INCEPTION V3 TRAINING FOR 90%+ F1-SCORE")
    print("="*70)
    
    # Load data
    print("\nLoading data with class balancing...")
    train_loader, val_loader, class_names, class_weights = get_data_loaders(config)
    
    # Create model
    print("Creating model...")
    model = create_inception_model(config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    
    # Setup weighted loss function
    weight_tensor = torch.FloatTensor([class_weights[i] for i in range(config.NUM_CLASSES)])
    weight_tensor = weight_tensor.to(config.DEVICE)
    criterion = FocalLoss(
        weight=weight_tensor,
        gamma=config.FOCAL_GAMMA,
        label_smoothing=config.LABEL_SMOOTHING
    )
    
    print(f"\nUsing weighted CrossEntropyLoss with weights: {weight_tensor.cpu().numpy()}")
    
    # Initialize history tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': [],
        'learning_rate_classifier': [],
        'learning_rate_backbone': [],
        'phase': []
    }
    
    # Add per-class tracking
    for cls in class_names:
        history[f'val_f1_{cls}'] = []
        history[f'val_precision_{cls}'] = []
        history[f'val_recall_{cls}'] = []
    
    best_f1 = 0.0
    epochs_without_improvement = 0
    global_epoch = 0
    
    print(f"\n{'='*70}")
    print("STARTING PROGRESSIVE TRAINING")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # PHASE 1: Train classifier only
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: TRAINING CLASSIFIER ONLY")
    print("="*70)
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
    def warmup_cosine(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return (epoch + 1) / config.WARMUP_EPOCHS
        progress = (epoch - config.WARMUP_EPOCHS) / (config.PHASE1_EPOCHS - config.WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    
    for epoch in range(config.PHASE1_EPOCHS):
        print(f"\nEpoch {global_epoch + 1} (Phase 1, Epoch {epoch + 1}/{config.PHASE1_EPOCHS})")
        print("-" * 70)
        
        # Train for one epoch
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, config, use_mixup=True
        )
        
        # Evaluate on validation set
        val_loss, val_metrics, val_tracker = evaluate(
            model, val_loader, criterion, config.DEVICE, config, use_tta=False
        )
        
        scheduler.step()
        
        # Save metrics to history
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
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f} | F1: {train_metrics['f1_macro']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | F1: {val_metrics['f1_macro']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        print(f"\nPer-class F1 scores:")
        for cls in class_names:
            print(f"  {cls}: {val_metrics[f'f1_{cls}']:.4f} (P: {val_metrics[f'precision_{cls}']:.3f}, R: {val_metrics[f'recall_{cls}']:.3f})")
        
        # Check for improvement
        if val_metrics['f1_macro'] > best_f1 + config.MIN_DELTA:
            best_f1 = val_metrics['f1_macro']
            epochs_without_improvement = 0
            
            # Save best model
            torch.save({
                'epoch': global_epoch,
                'phase': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'class_names': class_names
            }, os.path.join(config.OUTPUT_DIR, 'best_model.pth'))
            
            print(f"✓ New best model! F1: {best_f1:.4f}")
        else:
            epochs_without_improvement += 1
        
        global_epoch += 1
        
        # Early stopping
        if epochs_without_improvement >= config.PATIENCE:
            print(f"\nEarly stopping in Phase 1")
            break
    
    # ========================================================================
    # PHASE 2: Unfreeze last layers
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 2: UNFREEZING LAST LAYERS")
    print("="*70)
    
    # Load best model from Phase 1
    checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    unfreeze_layers(model, phase=2)
    
    param_groups = get_optimizer_groups(model, config, phase=2)
    optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    def warmup_cosine(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return (epoch + 1) / config.WARMUP_EPOCHS
        progress = (epoch - config.WARMUP_EPOCHS) / (config.PHASE1_EPOCHS - config.WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    
    epochs_without_improvement = 0
    
    for epoch in range(config.PHASE2_EPOCHS):
        print(f"\nEpoch {global_epoch + 1} (Phase 2, Epoch {epoch + 1}/{config.PHASE2_EPOCHS})")
        print("-" * 70)
        
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, config, use_mixup=True
        )
        
        val_loss, val_metrics, val_tracker = evaluate(
            model, val_loader, criterion, config.DEVICE, config, use_tta=False
        )
        
        scheduler.step()
        
        # Save metrics
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
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f} | F1: {train_metrics['f1_macro']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | F1: {val_metrics['f1_macro']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        print(f"\nPer-class F1 scores:")
        for cls in class_names:
            print(f"  {cls}: {val_metrics[f'f1_{cls}']:.4f} (P: {val_metrics[f'precision_{cls}']:.3f}, R: {val_metrics[f'recall_{cls}']:.3f})")
        
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
            
            print(f"✓ New best model! F1: {best_f1:.4f}")
        else:
            epochs_without_improvement += 1
        
        global_epoch += 1
        
        if epochs_without_improvement >= config.PATIENCE:
            print(f"\nEarly stopping in Phase 2")
            break
    
    # ========================================================================
    # PHASE 3: Fine-tune all layers (only if we haven't reached 90% yet)
    # ========================================================================
    if best_f1 < 0.90:
        print("\n" + "="*70)
        print("PHASE 3: FINE-TUNING ALL LAYERS")
        print("="*70)
        
        checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        unfreeze_layers(model, phase=3)
        
        param_groups = get_optimizer_groups(model, config, phase=3)
        optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
        def warmup_cosine(epoch):
            if epoch < config.WARMUP_EPOCHS:
                return (epoch + 1) / config.WARMUP_EPOCHS
            progress = (epoch - config.WARMUP_EPOCHS) / (config.PHASE1_EPOCHS - config.WARMUP_EPOCHS)
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)

        epochs_without_improvement = 0
        
        for epoch in range(config.PHASE3_EPOCHS):
            print(f"\nEpoch {global_epoch + 1} (Phase 3, Epoch {epoch + 1}/{config.PHASE3_EPOCHS})")
            print("-" * 70)
            
            # Less aggressive augmentation in phase 3
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, config.DEVICE, config, use_mixup=False
            )
            
            # Use TTA in phase 3 for better validation scores
            val_loss, val_metrics, val_tracker = evaluate(
                model, val_loader, criterion, config.DEVICE, config, use_tta=True
            )
            
            scheduler.step()
            
            # Save metrics
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
            
            print(f"Train Loss: {train_loss:.4f} | F1: {train_metrics['f1_macro']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | F1: {val_metrics['f1_macro']:.4f} | Acc: {val_metrics['accuracy']:.4f} (with TTA)")
            print(f"\nPer-class F1 scores:")
            for cls in class_names:
                print(f"  {cls}: {val_metrics[f'f1_{cls}']:.4f} (P: {val_metrics[f'precision_{cls}']:.3f}, R: {val_metrics[f'recall_{cls}']:.3f})")
            
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
                
                print(f"✓ New best model! F1: {best_f1:.4f}")
            else:
                epochs_without_improvement += 1
            
            global_epoch += 1
            
            # Stop if we reach target
            if best_f1 >= 0.90:
                print(f"\n🎉 ACHIEVED 90%+ F1-SCORE! Stopping training.")
                break
            
            if epochs_without_improvement >= config.PATIENCE:
                print(f"\nEarly stopping in Phase 3")
                break
    
    # ========================================================================
    # Final evaluation
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION WITH TEST-TIME AUGMENTATION")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate with TTA
    _, final_metrics, final_tracker = evaluate(
        model, val_loader, criterion, config.DEVICE, config, use_tta=True
    )
    
    # Print final results
    print(f"\nFINAL RESULTS:")
    print(f"{'='*70}")
    print(f"Accuracy:           {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"F1 Score (Macro):   {final_metrics['f1_macro']:.4f} ({final_metrics['f1_macro']*100:.2f}%)")
    print(f"F1 Score (Weighted): {final_metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro):  {final_metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):     {final_metrics['recall_macro']:.4f}")
    
    print(f"\nPer-Class Performance:")
    print(f"{'='*70}")
    for cls in class_names:
        f1 = final_metrics[f'f1_{cls}']
        prec = final_metrics[f'precision_{cls}']
        rec = final_metrics[f'recall_{cls}']
        status = "✓" if f1 >= 0.90 else "⚠"
        print(f"{status} {cls:10s}: F1={f1:.4f} ({f1*100:.2f}%), Precision={prec:.4f}, Recall={rec:.4f}")
    
    print(f"\n{final_tracker.get_classification_report()}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_history(history, config.OUTPUT_DIR)
    plot_confusion_matrix(final_tracker.get_confusion_matrix(), class_names, config.OUTPUT_DIR)
    plot_per_class_metrics(final_metrics, class_names, config.OUTPUT_DIR)
    
    # Save results to JSON
    with open(os.path.join(config.OUTPUT_DIR, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    with open(os.path.join(config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n✓ All results saved to {config.OUTPUT_DIR}")
    
    # Final status message
    if final_metrics['f1_macro'] >= 0.90:
        print(f"\n{'='*70}")
        print("🎉 SUCCESS! ACHIEVED 90%+ F1-SCORE TARGET! 🎉")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"⚠ Target not reached. Best F1: {final_metrics['f1_macro']:.4f} ({final_metrics['f1_macro']*100:.2f}%)")
        print("Consider: More data, ensemble methods, or different architecture")
        print(f"{'='*70}")
    
    return model, final_metrics, history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    config = Config()
    
    print("\n" + "="*70)
    print("OPTIMIZED INCEPTION V3 TRAINING CONFIGURATION")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max epochs: {config.NUM_EPOCHS}")
    print(f"Classifier LR: {config.CLASSIFIER_LR}")
    print(f"Backbone LR: {config.BACKBONE_LR}")
    print(f"MixUp alpha: {config.MIXUP_ALPHA}")
    print(f"CutMix prob: {config.CUTMIX_PROB}")
    print(f"Output: {config.OUTPUT_DIR}")
    print("="*70 + "\n")
    
    model, metrics, history = train_model(config)
    
    print("\n✓ Training complete!")