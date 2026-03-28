"""
Optuna hyperparameter search for Inception V3 VIPR classification.
Runs 50 trials, 40 epochs each, using Bayesian optimization.
Saves best trial config to optuna_best_inception.json
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import json
from collections import Counter
from sklearn.metrics import f1_score

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = './final_split_dataset'
INPUT_SIZE  = 299
NUM_CLASSES = 4
BATCH_SIZE  = 24
MAX_EPOCHS  = 60       # shorter per trial to save time
N_TRIALS    = 50
OUTPUT_DIR  = './results_inception_optuna'
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
def get_loaders(mixup_alpha, cutmix_prob):
    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    train_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
        transforms.RandomApply([transforms.GaussianBlur(5)], p=0.4),  # boost for Blurry
        transforms.ToTensor(), normalize,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(), normalize
    ])
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, 'validate'), transform=val_tf)

    targets = [l for _, l in train_ds.samples]
    counts  = Counter(targets)
    w_samp  = [1.0 / counts[l] for l in targets]
    sampler = WeightedRandomSampler(w_samp, len(w_samp), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    total = len(targets)
    class_w = {c: total / counts[c] for c in counts}
    max_w   = max(class_w.values())
    class_w = {c: v / max_w for c, v in class_w.items()}
    weight_tensor = torch.FloatTensor([class_w[i] for i in range(NUM_CLASSES)]).to(DEVICE)

    return train_loader, val_loader, weight_tensor


def mixup(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1-lam) * x[idx], y, y[idx], lam

def cutmix(x, y):
    lam = np.random.beta(1.0, 1.0)
    idx = torch.randperm(x.size(0)).to(x.device)
    W, H = x.size(2), x.size(3)
    rw, rh = int(W*np.sqrt(1-lam)), int(H*np.sqrt(1-lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1,y1 = max(cx-rw//2,0), max(cy-rh//2,0)
    x2,y2 = min(cx+rw//2,W), min(cy+rh//2,H)
    x[:,:,x1:x2,y1:y2] = x[idx,:,x1:x2,y1:y2]
    lam = 1 - (x2-x1)*(y2-y1)/(W*H)
    return x, y, y[idx], lam

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(dropout, hidden_size):
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model   = models.inception_v3(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    nf = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(nf, hidden_size),
        nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.BatchNorm1d(hidden_size // 2), nn.ReLU(), nn.Dropout(dropout * 0.8),
        nn.Linear(hidden_size // 2, NUM_CLASSES)
    )
    if hasattr(model, 'AuxLogits'):
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)
    # Unfreeze last block + classifier
    for name, p in model.named_parameters():
        if 'Mixed_7' in name or 'fc' in name or 'AuxLogits' in name:
            p.requires_grad = True
    return model.to(DEVICE)

# ── Training loop (one trial) ─────────────────────────────────────────────────
def run_trial(trial, train_loader, val_loader, weight_tensor, params):
    model    = build_model(params['dropout'], params['hidden_size'])
    criterion = FocalLoss(weight=weight_tensor,
                          gamma=params['focal_gamma'],
                          label_smoothing=params['label_smoothing'])
    optimizer = optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if p.requires_grad and 'fc' in n],
         'lr': params['lr']},
        {'params': [p for n,p in model.named_parameters() if p.requires_grad and 'fc' not in n],
         'lr': params['lr'] * params['backbone_lr_ratio']},
    ], weight_decay=params['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-7)
    best_f1   = 0.0

    for epoch in range(MAX_EPOCHS):
        # Mid-trial progressive unfreezing at epoch 20
        if epoch == 20:
            for name, p in model.named_parameters():
                if any(x in name for x in ['Mixed_6', 'Mixed_7', 'fc', 'AuxLogits']):
                    p.requires_grad = True
            # Update optimizer with newly unfrozen params
            classifier_params = [p for n,p in model.named_parameters()
                                 if p.requires_grad and ('fc' in n or 'AuxLogits' in n)]
            backbone_params   = [p for n,p in model.named_parameters()
                                 if p.requires_grad and 'fc' not in n and 'AuxLogits' not in n]
            optimizer.param_groups[0]['params'] = classifier_params
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['params'] = backbone_params
            else:
                optimizer.add_param_group({'params': backbone_params,
                                           'lr': params['lr'] * params['backbone_lr_ratio']})
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if np.random.rand() < 0.5:
                if np.random.rand() < params['cutmix_prob']:
                    inputs, ya, yb, lam = cutmix(inputs, labels)
                else:
                    inputs, ya, yb, lam = mixup(inputs, labels, params['mixup_alpha'])
                mixed = True
            else:
                mixed = False
            optimizer.zero_grad()
            out = model(inputs)
            main = out[0] if isinstance(out, tuple) else out
            if mixed:
                loss = lam * criterion(main, ya) + (1-lam) * criterion(main, yb)
                if isinstance(out, tuple):
                    loss += 0.4 * (lam * criterion(out[1], ya) + (1-lam) * criterion(out[1], yb))
            else:
                loss = criterion(main, labels)
                if isinstance(out, tuple):
                    loss += 0.4 * criterion(out[1], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                out    = model(inputs)
                preds  = torch.argmax(out, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        if val_f1 > best_f1:
            best_f1 = val_f1

        # Report to Optuna for pruning
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_f1

# ── Optuna Objective ──────────────────────────────────────────────────────────
def objective(trial):
    params = {
        'lr':               trial.suggest_float('lr',               5e-5, 2e-3, log=True),
        'weight_decay':     trial.suggest_float('weight_decay',     1e-5, 1e-1, log=True),
        'dropout':          trial.suggest_float('dropout',          0.2,  0.6),
        'focal_gamma':      trial.suggest_float('focal_gamma',      0.5,  3.5),
        'label_smoothing':  trial.suggest_float('label_smoothing',  0.0,  0.2),
        'backbone_lr_ratio':trial.suggest_float('backbone_lr_ratio',0.01, 0.3),
        'mixup_alpha':      trial.suggest_float('mixup_alpha',      0.1,  0.5),
        'cutmix_prob':      trial.suggest_float('cutmix_prob',      0.1,  0.6),
        'hidden_size':      trial.suggest_categorical('hidden_size',[512, 1024, 2048]),
    }
    train_loader, val_loader, weight_tensor = get_loaders(
        params['mixup_alpha'], params['cutmix_prob']
    )
    return run_trial(trial, train_loader, val_loader, weight_tensor, params)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=8, interval_steps=2),
        study_name='inception_vipr',
        storage=f'sqlite:///{OUTPUT_DIR}/inception_optuna.db',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=N_TRIALS, timeout=None)

    print('\n===== BEST TRIAL =====')
    best = study.best_trial
    print(f'F1: {best.value:.4f}')
    print('Params:')
    for k, v in best.params.items():
        print(f'  {k}: {v}')

    with open(f'{OUTPUT_DIR}/optuna_best_params.json', 'w') as f:
        json.dump({'f1': best.value, 'params': best.params}, f, indent=4)
    print(f'\nBest params saved to {OUTPUT_DIR}/optuna_best_params.json')