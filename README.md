# Arnold-Lab-VIPR-Undergraduate-Classification

A config-driven deep learning framework for benchmarking pretrained CNN/Transformer backbones on a 4-class image classification task. Built for the VIPR undergraduate project at Arnold Lab (UGA) under Shufan as mentor.

## Table of Contents

- [Setup on GACRC](#setup-on-gacrc)
- [Running Experiments](#running-experiments)
  - [Single Experiment](#1-single-experiment)
  - [Batch Experiments](#2-batch-experiments-multiple-models)
  - [Optuna Hyperparameter Optimization](#3-optuna-hyperparameter-optimization)
- [Configuration Reference](#configuration-reference)
  - [Standard Training Config](#standard-training-config)
  - [Progressive Unfreezing Config](#progressive-unfreezing-thaw-config)
  - [Dynamic Unfreezing Config](#dynamic-unfreezing-config)
  - [Optuna Search Config](#optuna-search-config)
- [Supported Backbones](#supported-backbones)
- [Dataset Format](#dataset-format)
- [Results Structure](#results-structure)
- [Dataset Audit Tool](#dataset-audit-tool)
- [Sigmoid Granular Layer Sweeps](#sigmoid-granular-layer-sweeps)

---

## Setup on GACRC

### 1. Create Virtual Environment in Scratch

Virtual environments should be created in `/scratch` for better performance on the cluster:

```bash
# Navigate to your scratch directory
cd /scratch/$USER

# Create a virtual environment
module load Python/3.10.4-GCCcore-11.3.0
python -m venv scratch_venv

# Activate the environment
source /scratch/$USER/scratch_venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
pip install optuna pandas matplotlib numpy
```

### 3. Clone Repository and Copy Data

```bash
# Clone the repository
git clone <repository-url>
cd Arnold-Lab-VIPR-Undergraduate-Classification

# Copy dataset from shared location
cp -r /work/OMLPA/HL98745/final_split_dataset /scratch/$USER/data/
```

### 4. Update Config Paths

Update the `data.path` in your config files to point to your data location:
```json
"data": {
    "path": "/scratch/<your_username>/data/final_split_dataset",
    ...
}
```

---

## Running Experiments

There are three levels of scripting for running experiments:

### 1. Single Experiment

Run a single model training with one configuration file.

**Direct Python:**
```bash
python run_experiment.py configs/resnet152.json
```

**Via SLURM:**
```bash
sbatch run_experiment.slurm configs/resnet152.json
```

The SLURM script allocates: 1x A100 GPU, 8 CPUs, 64GB RAM, 12 hours.

---

### 2. Batch Experiments (Multiple Models)

Run multiple models sequentially with separate config files.

**Direct Python:**
```bash
python run_batch.py configs/resnet152.json configs/vgg16.json configs/swin_t.json
```

**Using wildcard:**
```bash
python run_batch.py configs/*.json
```

**Via SLURM (runs all 7 baseline models):**
```bash
sbatch run_batch.slurm
```

The batch SLURM script allocates: 1x A100 GPU, 8 CPUs, 64GB RAM, 48 hours.

---

### 3. Optuna Hyperparameter Optimization

Run automated hyperparameter search using Bayesian optimization.

**Direct Python:**
```bash
python run_optuna.py configs/optuna/resnet152_sweep.json
```

**Resume an interrupted study:**
```bash
python run_optuna.py configs/optuna/resnet152_sweep.json --resume
```

**Via SLURM:**
```bash
# New study
sbatch run_optuna.slurm configs/optuna/resnet152_sweep.json

# Resume existing study
sbatch run_optuna.slurm configs/optuna/resnet152_sweep.json --resume
```

The Optuna SLURM script allocates: 1x A100 GPU, 8 CPUs, 64GB RAM, 48 hours.

---

## Configuration Reference

### Standard Training Config

Used for single experiments with `run_experiment.py` or `run_batch.py`.

**Example:** `configs/resnet152.json`

```json
{
    "experiment_name": "resnet152_frozen",

    "model": {
        "backbone": "resnet152",
        "pretrained": true,
        "freeze_backbone": true,
        "classifier_hidden": [1024],
        "dropout": 0.2
    },

    "data": {
        "path": "/scratch/hl98745/data/final_split_dataset",
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

#### Model Options

| Option | Type | Description |
|--------|------|-------------|
| `backbone` | string | Architecture name (see [Supported Backbones](#supported-backbones)) |
| `pretrained` | bool | Load ImageNet pre-trained weights |
| `freeze_backbone` | bool | Freeze backbone parameters (only train classifier) |
| `classifier_hidden` | list[int] | Hidden layer sizes for classifier head. Empty `[]` = linear classifier |
| `dropout` | float | Dropout probability in classifier (0.0 - 1.0) |

#### Data Options

| Option | Type | Description |
|--------|------|-------------|
| `path` | string | Root path to dataset (must contain `train/`, `validate/`, `test/` folders) |
| `input_size` | int | Target image size (images resized to input_size x input_size) |
| `batch_size` | int | Training batch size |
| `num_workers` | int | DataLoader worker processes |

#### Augmentation Options

| Option | Type | Description |
|--------|------|-------------|
| `horizontal_flip` | bool | Random horizontal flip (p=0.5) |
| `vertical_flip` | bool | Random vertical flip (p=0.5) |
| `random_rotation` | float | Maximum rotation degrees (0 = disabled) |
| `color_jitter` | float | Color jitter intensity (0 = disabled) |
| `random_crop_scale` | [float, float] | Scale range for RandomResizedCrop |

#### Training Options

| Option | Type | Description |
|--------|------|-------------|
| `epochs` | int | Maximum training epochs |
| `learning_rate` | float | Base learning rate |
| `optimizer` | string | Optimizer: `"adamw"`, `"adam"`, or `"sgd"` |
| `weight_decay` | float | L2 regularization coefficient |
| `scheduler` | string | LR scheduler: `"cosine"`, `"step"`, or `null` |
| `seed` | int | Random seed for reproducibility |
| `early_stopping_patience` | int | Stop if val_acc doesn't improve for N epochs |

---

### Progressive Unfreezing (Thaw) Config

Extends standard config with backbone unfreezing schedule for fine-tuning.

**Example:** `configs/resnet152_thaw.json`

```json
{
    "experiment_name": "resnet152_progressive_thaw",

    "model": {
        "backbone": "resnet152",
        "pretrained": true,
        "freeze_backbone": true,
        "classifier_hidden": [1024],
        "dropout": 0.2
    },

    "data": { ... },
    "augmentations": { ... },

    "training": {
        "epochs": 100,
        "learning_rate": 0.0001,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "seed": 42,
        "early_stopping_patience": 20,

        "thaw_schedule": {
            "20": 0.2,
            "40": 0.5,
            "60": 1.0
        },
        "backbone_lr_ratio": 0.1
    }
}
```

#### Thaw-Specific Options

| Option | Type | Description |
|--------|------|-------------|
| `thaw_schedule` | dict | Map of epoch → percentage of backbone to unfreeze. Keys are epoch numbers (as strings), values are percentages (0.0-1.0). Layers unfreeze from output toward input. |
| `backbone_lr_ratio` | float | Learning rate multiplier for unfrozen backbone layers. E.g., `0.1` means backbone learns at 10% of the classifier learning rate. |

**How Thawing Works:**
1. Training starts with backbone frozen (only classifier trainable)
2. At specified epochs, backbone layers progressively unfreeze from output to input
3. Unfrozen backbone layers use a lower learning rate (controlled by `backbone_lr_ratio`)
4. This prevents catastrophic forgetting while allowing fine-tuning

---

### Dynamic Unfreezing Config

Extends the standard config with **plateau-triggered** automatic backbone thawing. Unlike `thaw_schedule` (which unfreezes at fixed epochs), dynamic unfreezing monitors validation loss and thaws the next group of backbone layers only when progress stalls.

**Example:** `configs/resnet152_dynamic_unfreeze.json`

```json
{
    "experiment_name": "resnet152_dynamic_unfreeze",

    "model": {
        "backbone": "resnet152",
        "pretrained": true,
        "freeze_backbone": true,
        "classifier_hidden": [2048, 1024],
        "dropout": 0.32
    },

    "data": { ... },
    "augmentations": { ... },

    "training": {
        "epochs": 100,
        "learning_rate": 0.0006,
        "optimizer": "adamw",
        "weight_decay": 3e-5,
        "scheduler": "cosine",
        "seed": 42,
        "early_stopping_patience": 20,

        "dynamic_unfreeze": {
            "unfreeze_patience": 8,
            "unfreeze_size": 1,
            "lr_decay_ratio": 0.1
        }
    }
}
```

#### Dynamic Unfreezing Options

| Option | Type | Description |
|--------|------|-------------|
| `unfreeze_patience` | int | Epochs with no val-loss improvement before thawing the next unit |
| `unfreeze_size` | int | Number of backbone units to thaw per trigger event |
| `lr_decay_ratio` | float | LR multiplier per depth level — deeper units use `base_lr × ratio^depth` |

**How Dynamic Unfreezing Works:**
1. Training starts with the backbone fully frozen (only classifier trains).
2. `DynamicThawController` tracks validation loss; when improvement stalls for `unfreeze_patience` epochs, it thaws the next `unfreeze_size` units moving from output layers toward input.
3. Each newly unfrozen group receives a decayed learning rate (`base_lr × lr_decay_ratio^depth`), preventing catastrophic forgetting of early features.
4. Compatible with all CNN backbones (ResNet, DenseNet, VGG, EfficientNet, etc.) and Vision Transformers (ViT, Swin).

> **Note:** `thaw_schedule` and `dynamic_unfreeze` are mutually exclusive. If both keys are present, `dynamic_unfreeze` takes precedence.

---

### Optuna Search Config

Defines the hyperparameter search space for Bayesian optimization.

**Example:** `configs/optuna/resnet152_sweep.json`

```json
{
    "study_name": "resnet152_fine_tuning_sweep",
    "base_config_path": "configs/resnet152_thaw.json",

    "n_trials": 50,
    "timeout_hours": 24,

    "pruning": {
        "enabled": true,
        "n_startup_trials": 5,
        "n_warmup_steps": 10,
        "interval_steps": 2
    },

    "cleanup": {
        "keep_top_n": 5,
        "cleanup_frequency": 10
    },

    "search_space": {
        "training.learning_rate": {
            "type": "log_float",
            "low": 1e-5,
            "high": 1e-3
        },
        "training.backbone_lr_ratio": {
            "type": "float",
            "low": 0.01,
            "high": 0.1
        },
        "training.weight_decay": {
            "type": "log_float",
            "low": 1e-5,
            "high": 1e-2
        },
        "training.thaw_epoch": {
            "type": "int",
            "low": 5,
            "high": 25
        },
        "training.thaw_percent": {
            "type": "categorical",
            "choices": [0.25, 0.5, 0.75, 1.0]
        },
        "augmentations.color_jitter": {
            "type": "float",
            "low": 0.0,
            "high": 0.15
        },
        "augmentations.random_rotation": {
            "type": "float",
            "low": 0.0,
            "high": 180.0
        },
        "model.classifier_hidden": {
            "type": "categorical",
            "choices": [[512], [1024], [1024, 512]]
        },
        "model.dropout": {
            "type": "float",
            "low": 0.1,
            "high": 0.4
        }
    }
}
```

#### Optuna Study Options

| Option | Type | Description |
|--------|------|-------------|
| `study_name` | string | Unique identifier for this optimization study |
| `base_config_path` | string | Path to base training config that will be modified by search |
| `n_trials` | int | Number of hyperparameter combinations to try |
| `timeout_hours` | float/null | Maximum runtime in hours (`null` = unlimited) |

#### Pruning Options

| Option | Type | Description |
|--------|------|-------------|
| `enabled` | bool | Enable early stopping of unpromising trials |
| `n_startup_trials` | int | Number of trials to run before pruning starts |
| `n_warmup_steps` | int | Epochs to train before pruning checks begin |
| `interval_steps` | int | Check pruning every N epochs |

#### Cleanup Options

| Option | Type | Description |
|--------|------|-------------|
| `keep_top_n` | int | Number of best trial models to keep (others deleted to save space) |
| `cleanup_frequency` | int | Run cleanup every N trials |

#### Search Space Types

Each parameter in `search_space` uses dot notation (e.g., `training.learning_rate`) and requires. Nested keys work too — for example, sweeping dynamic unfreezing hyperparameters:

```json
"search_space": {
    "training.dynamic_unfreeze.lr_decay_ratio": {
        "type": "log_float",
        "low": 0.01,
        "high": 0.5
    },
    "training.dynamic_unfreeze.unfreeze_patience": {
        "type": "int",
        "low": 3,
        "high": 20
    }
}
```

See `configs/optuna/resnet152_gradual_unfreeze_sweep.json` for a working example.

##### Type Reference

| Type | Parameters | Description |
|------|------------|-------------|
| `log_float` | `low`, `high` | Float sampled on log scale (best for learning rates) |
| `float` | `low`, `high` | Float sampled on linear scale |
| `int` | `low`, `high` | Integer in range [low, high] |
| `categorical` | `choices` | Pick from list of discrete values |
| `hidden_layers` | `n_layers`, `size_low`, `size_high` | Variable-depth MLP head search (see below) |

##### `hidden_layers` Type

Searches both the number of hidden layers and each layer's size as a continuous integer range. Replaces the hardcoded `categorical` list approach.

```json
"model.classifier_hidden": {
    "type": "hidden_layers",
    "n_layers": [1, 2],
    "size_low": 128,
    "size_high": 2048,
    "size_step": 1,
    "size_log": false,
    "enforce_decreasing": false
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `n_layers` | required | List of valid layer counts, e.g. `[1, 2]` |
| `size_low` / `size_high` | required | Integer range for each layer size |
| `size_step` | `1` | Quantize sizes to multiples of this value |
| `size_log` | `false` | Sample sizes on log scale |
| `enforce_decreasing` | `false` | Sort layer sizes descending (funnel architecture) |

Optuna registers `n_layers` as categorical and each `layer_{i}_size` as an integer parameter. All layer sizes are always sampled (up to `max(n_layers)`) to avoid conditional-parameter issues with the TPE sampler; the result is then sliced to the sampled depth.

---

## Supported Backbones

### ResNet Family
- `resnet50`, `resnet101`, `resnet152`

### DenseNet Family
- `densenet121`, `densenet169`, `densenet201`

### VGG Family
- `vgg16`, `vgg19`

### Classic CNNs
- `alexnet`
- `inception_v3`, `googlenet`

### EfficientNet Family
- `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`

### Vision Transformers
- `vit_b_16`, `vit_b_32`, `vit_l_16`

### Swin Transformers
- `swin_t`, `swin_s`, `swin_b`

### ConvNeXt Family
- `convnext_tiny`, `convnext_small`, `convnext_base`

---

## Dataset Format

The dataset must follow ImageFolder structure with `train/`, `validate/`, and `test/` splits:

```
final_split_dataset/
├── train/
│   ├── Blurry/
│   │   ├── img_001.jpg
│   │   └── ...
│   ├── Good/
│   ├── Opaque/
│   └── Yellow/
├── validate/
│   └── (same structure)
└── test/
    └── (same structure)
```

Classes are determined alphabetically from folder names.

---

## Results Structure

### Single Experiment Output

```
results/<experiment_name>/
├── config.json              # Copy of input configuration
├── best_model.pth           # Best model weights (by val_acc)
├── metrics.csv              # Training metrics per epoch
├── results.json             # Final summary (test accuracy, etc.)
├── loss_curves.png          # Training/validation loss plot
├── accuracy_curves.png      # Accuracy over epochs
├── learning_rate.png        # LR schedule visualization
└── experiment_summary.png   # Combined visualization
```

### Optuna Study Output

```
results/optuna_studies/<study_name>/
├── study.db                 # SQLite database (persistent, resumable)
├── optuna_config.json       # Copy of Optuna config
├── study_summary.json       # Best trial results
├── best/                    # Best trial model
│   └── <trial_results>/
└── trials/
    ├── <study_name>_trial0_xxx/
    │   ├── metrics.csv
    │   ├── results.json
    │   └── best_model.pth
    └── ...
```

---

## Monitoring Jobs on GACRC

```bash
# Check job status
squeue -u $USER

# Watch job output in real-time
tail -f vipr_train_*.out

# Cancel a job
scancel <job_id>
```

---

## Dataset Audit Tool

`audit_dataset.py` loads a trained model from any experiment results directory, runs inference over a dataset split, and saves every misclassified image into an organised folder structure for manual inspection.

### Usage

```bash
# Audit the validate split (default)
python audit_dataset.py results/resnet152_frozen

# Audit the test split
python audit_dataset.py results/resnet152_frozen --split test

# Override the data path (useful when running locally with a different data location)
python audit_dataset.py results/resnet152_frozen --split test --data-path /local/path/to/data

# Audit an Optuna best trial
python audit_dataset.py results/optuna_studies/resnet152_fine_tuning_sweep/best
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `results_dir` | (required) | Path to the experiment results directory containing `config.json` and `best_model.pth` |
| `--split` | `validate` | Dataset split to evaluate: `train`, `validate`, or `test` |
| `--output` | `dataset_audit` | Output directory for misclassified images and summary file |
| `--data-path` | (from config) | Override the data root from `config.json` |

### Output Structure

```
dataset_audit/
├── Blurry_as_Good/
│   ├── 9421_img_001.jpg    ← confidence prefix in basis points (9421 = 94.21%)
│   └── ...
├── Good_as_Opaque/
│   └── ...
└── audit_summary.txt       ← overall accuracy, per-class accuracy, bucket counts
```

Each misclassified image is copied into a `{CorrectClass}_as_{GuessedClass}/` folder. The filename is prefixed with a 4-digit basis-point confidence score so images can be sorted by model certainty.

---

## Sigmoid Granular Layer Sweeps

A set of pre-built Optuna sweep experiments using `bce_sigmoid` loss with **fully frozen backbones** and the new `hidden_layers` search space (1–2 layers, sizes 128–2048).

### Config locations

| Directory | Contents |
|---|---|
| `configs/base_config_sigmoid/` | Base training configs for each backbone (sigmoid, frozen, new data path) |
| `configs/sigmoid_granular_layers/` | Optuna sweep configs referencing the above |

### Backbones covered

alexnet, convnext_base, densenet201, efficientnet_b4, inception_v3, resnet152, swin_b, vgg16, vgg19, vit_l_16

### Submit all jobs at once

```bash
bash submit_sigmoid_granular_sweeps.sh
```

Results land in `results/optuna_studies/sigmoid_granular/<backbone>_sigmoid_frozen_granular/`.
