# VIPR Classification Framework - Design Document

## Overview

A minimal, config-driven framework for benchmarking pretrained CNN/Transformer backbones on a 4-class image classification task. The system prioritizes simplicity and reproducibility.

---

## File Structure

```
project/
├── utils/
│   ├── __init__.py       # Package exports
│   ├── model_utils.py    # Backbone loading + classifier construction
│   └── trainer.py        # Training loop, metrics, exports train_model()
├── configs/
│   └── *.json            # Experiment configurations
├── results/
│   └── [experiment_name]/
│       ├── best_model.pth
│       ├── metrics.csv
│       ├── results.json  # Final summary
│       └── config.json   # Copy of config used
└── run_experiment.py     # Entry point (thin wrapper)
```

**Total core files: 2** (`model_utils.py`, `trainer.py`)

---

## Configuration Schema (options.json)

```json
{
    "experiment_name": "resnet152_baseline_v1",
    "model": {
        "backbone": "resnet152",
        "pretrained": true,
        "freeze_backbone": true,
        "classifier_hidden": [1024],
        "dropout": 0.2
    },
    "data": {
        "path": "/path/to/final_split_dataset",
        "input_size": 224,
        "batch_size": 32,
        "num_workers": 4
    },
    "augmentations": {
        "horizontal_flip": true,
        "vertical_flip": false,
        "random_rotation": 15,
        "color_jitter": 0.1,
        "random_crop_scale": [0.8, 1.0]
    },
    "training": {
        "epochs": 50,
        "learning_rate": 0.0001,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "seed": 42,
        "early_stopping_patience": 20
    }
}
```

### Schema Details

| Field | Type | Description |
|-------|------|-------------|
| `experiment_name` | string | Unique ID for results folder |
| `model.backbone` | string | Any torchvision model name (e.g., `resnet152`, `densenet201`, `swin_t`, `efficientnet_b0`) |
| `model.pretrained` | bool | Load ImageNet weights |
| `model.freeze_backbone` | bool | Freeze all backbone parameters (Phase 1 = true) |
| `model.classifier_hidden` | int[] | Hidden layer sizes. Output (4) is appended automatically. Feature input is auto-detected. |
| `model.dropout` | float | Dropout before final layer |
| `data.input_size` | int | Resize target. Model-specific (224 for ResNet, 384 for some ViTs) |
| `data.path` | string | Root folder containing `train/`, `val/`, `test/` subfolders |
| `augmentations.*` | mixed | See Augmentation Strategy below |
| `training.scheduler` | string | `"cosine"`, `"step"`, or `null` for constant LR |
| `training.early_stopping_patience` | int | Stop if no improvement for N epochs (default: 20) |

---

## Module Specifications

### 1. model_utils.py

**Purpose**: Load backbone, auto-detect feature size, attach custom classifier.

```python
# Public API

def load_model(options: dict) -> torch.nn.Module:
    """
    Build complete model from config.

    Args:
        options: Full config dict (uses options['model'] internally)

    Returns:
        Model with frozen/unfrozen backbone + custom classifier head

    Example:
        model = load_model(config)
        # model.backbone_features = 2048 (for resnet152)
        # model.classifier = Sequential(Linear(2048, 1024), ReLU, Dropout, Linear(1024, 4))
    """

def get_feature_dim(model: torch.nn.Module, backbone_name: str) -> int:
    """
    Auto-detect the output feature dimension of a backbone.

    Handles different architectures:
    - ResNet/DenseNet: model.fc.in_features or model.classifier.in_features
    - EfficientNet: model.classifier[1].in_features
    - Swin/ViT: model.head.in_features

    Returns:
        int: Feature dimension (e.g., 2048 for ResNet152)
    """
```

**Classifier Construction Logic**:
```
Input:  classifier_hidden = [1024], dropout = 0.2, num_classes = 4
        feature_dim = 2048 (auto-detected)

Output: nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 4)
        )
```

**Supported Backbones** (initial list):
- ResNet: `resnet50`, `resnet101`, `resnet152`
- DenseNet: `densenet121`, `densenet201`
- EfficientNet: `efficientnet_b0`, `efficientnet_b4`
- Vision Transformer: `vit_b_16`, `vit_l_16`
- Swin: `swin_t`, `swin_s`, `swin_b`
- ConvNeXt: `convnext_tiny`, `convnext_base`

---

### 2. trainer.py

**Purpose**: Stateless training function. Takes config, runs experiment, saves results.

```python
# Public API

def train_model(options: dict) -> dict:
    """
    Run a complete training experiment.

    Args:
        options: Full config dict

    Returns:
        dict: {
            'best_val_acc_top1': float,
            'best_val_acc_top3': float,
            'best_epoch': int,
            'final_test_acc_top1': float,
            'final_test_acc_top3': float,
            'inference_time_ms': float,
            'total_params': int,
            'trainable_params': int
        }

    Side Effects:
        - Creates results/[experiment_name]/ folder
        - Saves best_model.pth, metrics.csv, config.json
    """

def evaluate_model(model, dataloader, device) -> dict:
    """
    Evaluate model on a dataloader.

    Returns:
        dict: {'loss': float, 'acc_top1': float, 'acc_top3': float}
    """

def measure_inference_time(model, input_size, device, iterations=100) -> float:
    """
    Average inference time in milliseconds.
    """
```

**Training Loop Pseudocode**:
```
1. Set seed (options['training']['seed'])
2. Create experiment folder: results/{experiment_name}/
3. Save config.json (copy of input options)
4. Build dataloaders with augmentations
5. Load model via model_utils.load_model()
6. Setup optimizer, scheduler, loss (CrossEntropy)
7. For each epoch:
   a. Train one epoch, record loss
   b. Validate, record top-1 and top-3 accuracy
   c. If best val_acc_top1 → save best_model.pth
   d. Append row to metrics.csv
8. Load best_model.pth
9. Evaluate on test set
10. Measure inference time
11. Return summary dict
```

**metrics.csv Format**:
```csv
epoch,train_loss,val_loss,val_acc_top1,val_acc_top3,lr
1,2.145,1.832,0.421,0.876,0.0001
2,1.654,1.423,0.534,0.912,0.0001
...
```

---

## Augmentation Strategy

Augmentations are built from the config dict. Each key maps to a specific transform.

| Config Key | Transform | Value Meaning |
|------------|-----------|---------------|
| `horizontal_flip: true` | RandomHorizontalFlip(p=0.5) | Enable/disable |
| `vertical_flip: true` | RandomVerticalFlip(p=0.5) | Enable/disable |
| `random_rotation: 15` | RandomRotation(15) | Max degrees (0 = disabled) |
| `color_jitter: 0.1` | ColorJitter(0.1, 0.1, 0.1, 0.05) | Intensity (0 = disabled) |
| `random_crop_scale: [0.8, 1.0]` | RandomResizedCrop(input_size, scale) | Scale range (null = disabled) |

**Transform Pipeline**:
```python
# Training
transforms.Compose([
    # Augmentations (from config)
    transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),  # if enabled
    transforms.RandomHorizontalFlip(),  # if enabled
    transforms.RandomRotation(15),  # if enabled
    transforms.ColorJitter(...),  # if enabled
    # Standard normalization
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test (no augmentation)
transforms.Compose([
    transforms.Resize(int(input_size * 1.14)),  # Slight upscale
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## Data Expectations

```
final_split_dataset/
├── train/
│   ├── class_0/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── class_1/
│   ├── class_2/
│   └── class_3/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

Uses `torchvision.datasets.ImageFolder` - class names derived from folder names.

---

## Usage Examples

### Single Experiment
```python
from utils.trainer import train_model
import json

with open('configs/resnet152_baseline.json') as f:
    config = json.load(f)

results = train_model(config)
print(f"Test Accuracy: {results['final_test_acc_top1']:.2%}")
```

### Hyperparameter Search
```python
from utils.trainer import train_model
import json

backbones = ['resnet152', 'densenet201', 'swin_t', 'efficientnet_b4']
input_sizes = {'resnet152': 224, 'densenet201': 224, 'swin_t': 224, 'efficientnet_b4': 380}

results_summary = []

for backbone in backbones:
    with open('configs/base_config.json') as f:
        config = json.load(f)

    config['experiment_name'] = f'{backbone}_frozen_v1'
    config['model']['backbone'] = backbone
    config['data']['input_size'] = input_sizes[backbone]

    result = train_model(config)
    result['backbone'] = backbone
    results_summary.append(result)

# Sort by top-1 accuracy
results_summary.sort(key=lambda x: x['final_test_acc_top1'], reverse=True)
```

---

## Implementation Decisions

| Decision | Choice |
|----------|--------|
| Mixed Precision | Enabled via `torch.cuda.amp` |
| Early Stopping | Enabled with configurable patience (default: 20 epochs) |
| Class Imbalance | Not needed (balanced dataset) |
| Logging | Console output only |
| Test Evaluation | End of training only; validation used during training |

---

## Phase 1 Checklist

- [x] Implement `model_utils.py`
- [x] Implement `trainer.py`
- [x] Create base config JSON
- [ ] Test with 1 backbone (ResNet152)
- [ ] Run benchmark across 10+ backbones
- [ ] Generate comparison table (accuracy vs inference time)

## Phase 2 (Future)

- Add `freeze_backbone: false` support
- Implement progressive unfreezing schedule
- Fine-tune top 3 backbones from Phase 1
