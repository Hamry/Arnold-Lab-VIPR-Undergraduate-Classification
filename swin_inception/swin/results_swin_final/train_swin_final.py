"""
Swin Transformer FINAL training using best Optuna hyperparameters.
Best Optuna F1: 89.53% — this script runs full 100 epochs to push to 95%+
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import json
from collections import Counter

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
        return (((1-pt)**self.gamma)*ce).mean()

class Config:
    DATA_DIR        = './final_split_dataset'
    TRAIN_DIR       = 'train'
    VAL_DIR         = 'validate'
    INPUT_SIZE      = 224
    NUM_CLASSES     = 4
    BATCH_SIZE      = 24
    PHASE1_EPOCHS   = 35
    PHASE2_EPOCHS   = 40
    PHASE3_EPOCHS   = 50
    PATIENCE        = 25
    MIN_DELTA       = 0.0005
    OUTPUT_DIR      = './results_swin_final_v2'
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── BEST OPTUNA PARAMS (2nd run Trial 23) ─────────────────────────────────
    LR              = 0.00048405480969545563
    WEIGHT_DECAY    = 0.043211958008344785
    DROPOUT         = 0.5571398437847817
    FOCAL_GAMMA     = 1.83113749838529
    LABEL_SMOOTHING = 0.11155954800699658
    BACKBONE_LR_RATIO = 0.1996744699657122
    MIXUP_ALPHA     = 0.3371528676561022
    CUTMIX_PROB     = 0.25568269163734547
    HIDDEN_SIZE     = 2048

def get_class_weights(dataset):
    targets = [label for _, label in dataset.samples]
    counts  = Counter(targets)
    total   = len(targets)
    weights = {c: total / counts[c] for c in counts}
    max_w   = max(weights.values())
    weights = {c: v/max_w for c,v in weights.items()}
    blurry_idx = dataset.classes.index('Blurry')
    weights[blurry_idx] *= 2.0
    return weights

def get_balanced_sampler(dataset):
    targets = [label for _, label in dataset.samples]
    counts  = Counter(targets)
    w       = [1.0/counts[l] for l in targets]
    return WeightedRandomSampler(w, len(w), replacement=True)

def mixup(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

def cutmix(x, y):
    lam = np.random.beta(1.0, 1.0)
    idx = torch.randperm(x.size(0)).to(x.device)
    W, H = x.size(2), x.size(3)
    rw = int(W*np.sqrt(1-lam)); rh = int(H*np.sqrt(1-lam))
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1,y1 = max(cx-rw//2,0), max(cy-rh//2,0)
    x2,y2 = min(cx+rw//2,W), min(cy+rh//2,H)
    x[:,:,x1:x2,y1:y2] = x[idx,:,x1:x2,y1:y2]
    lam = 1-(x2-x1)*(y2-y1)/(W*H)
    return x, y, y[idx], lam

def get_loaders(config):
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    train_tf = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        transforms.RandomResizedCrop(config.INPUT_SIZE, scale=(0.6,1.0)),
        transforms.RandomAffine(degrees=15, translate=(0.15,0.15), scale=(0.9,1.1)),
        transforms.RandAugment(num_ops=3, magnitude=12),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.5,3.0))], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(9, sigma=(1.0,5.0))], p=0.3),
        transforms.ToTensor(), norm,
        transforms.RandomErasing(p=0.25, scale=(0.02,0.2)),
    ])
    val_tf = transforms.Compose([transforms.Resize((config.INPUT_SIZE,config.INPUT_SIZE)), transforms.ToTensor(), norm])
    train_ds = datasets.ImageFolder(os.path.join(config.DATA_DIR, config.TRAIN_DIR), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(config.DATA_DIR, config.VAL_DIR),   transform=val_tf)
    cw      = get_class_weights(train_ds)
    sampler = get_balanced_sampler(train_ds)
    wt      = torch.FloatTensor([cw[i] for i in range(config.NUM_CLASSES)]).to(config.DEVICE)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False,   num_workers=4, pin_memory=True)
    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"Weights: {[round(cw[i],3) for i in range(config.NUM_CLASSES)]}")
    return train_loader, val_loader, train_ds.classes, wt

def build_model(config):
    model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False
    nf = model.head.in_features
    h  = config.HIDDEN_SIZE
    model.head = nn.Sequential(
        nn.Linear(nf, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(config.DROPOUT),
        nn.Linear(h, h//2), nn.LayerNorm(h//2), nn.GELU(), nn.Dropout(config.DROPOUT*0.75),
        nn.Linear(h//2, config.NUM_CLASSES)
    )
    return model.to(config.DEVICE)

def unfreeze(model, phase):
    if phase == 1:
        for p in model.parameters(): p.requires_grad = False
        for p in model.head.parameters(): p.requires_grad = True
        print("Phase 1: head only")
    elif phase == 2:
        for n,p in model.named_parameters():
            if any(x in n for x in ['features.5','features.6','features.7','head','norm']):
                p.requires_grad = True
        print("Phase 2: features.5+6+7 + head")
    elif phase == 3:
        for p in model.parameters(): p.requires_grad = True
        print("Phase 3: all layers")
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {tr:,}/{tt:,} ({100*tr/tt:.1f}%)")

def make_optimizer(model, config, phase):
    cls_p  = [p for n,p in model.named_parameters() if p.requires_grad and 'head' in n]
    back_p = [p for n,p in model.named_parameters() if p.requires_grad and 'head' not in n]
    if phase == 1:
        return optim.AdamW(cls_p, lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    return optim.AdamW([
        {'params': cls_p,  'lr': config.LR},
        {'params': back_p, 'lr': config.LR * config.BACKBONE_LR_RATIO},
    ], weight_decay=config.WEIGHT_DECAY)

def train_epoch(model, loader, criterion, optimizer, config, use_aug=True):
    model.train()
    loss_sum, preds_all, labels_all = 0, [], []
    for x, y in loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        if use_aug and np.random.rand() < 0.5:
            if np.random.rand() < config.CUTMIX_PROB:
                x, ya, yb, lam = cutmix(x, y)
            else:
                x, ya, yb, lam = mixup(x, y, config.MIXUP_ALPHA)
            mixed = True
        else:
            mixed = False
        optimizer.zero_grad()
        out  = model(x)
        loss = (lam*criterion(out,ya)+(1-lam)*criterion(out,yb)) if mixed else criterion(out,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()*x.size(0)
        preds_all.extend(torch.argmax(out,1).cpu().numpy())
        labels_all.extend((ya if mixed else y).cpu().numpy())
    return loss_sum/len(loader.dataset), f1_score(labels_all, preds_all, average='macro')

def evaluate(model, loader, criterion, config, tta=False):
    model.eval()
    loss_sum, preds_all, labels_all = 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            if tta:
                ps = [torch.softmax(model(x),1),
                      torch.softmax(model(torch.flip(x,[3])),1),
                      torch.softmax(model(torch.flip(x,[2])),1),
                      torch.softmax(model(torch.flip(x,[2,3])),1),
                      torch.softmax(model(torch.rot90(x,1,[2,3])),1),
                      torch.softmax(model(torch.rot90(x,3,[2,3])),1)]
                probs = torch.stack(ps).mean(0)
                out   = torch.log(probs)
            else:
                out   = model(x)
                probs = torch.softmax(out,1)
            loss_sum += criterion(out,y).item()*x.size(0)
            preds_all.extend(torch.argmax(probs,1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())
    val_f1 = f1_score(labels_all, preds_all, average='macro')
    classes = loader.dataset.classes
    metrics = {
        'f1_macro': val_f1,
        'accuracy': accuracy_score(labels_all, preds_all),
        'f1_weighted': f1_score(labels_all, preds_all, average='weighted'),
        'precision_macro': precision_score(labels_all, preds_all, average='macro', zero_division=0),
        'recall_macro': recall_score(labels_all, preds_all, average='macro', zero_division=0),
    }
    for i,c in enumerate(classes):
        f1c = f1_score(labels_all, preds_all, average=None, zero_division=0)
        pc  = precision_score(labels_all, preds_all, average=None, zero_division=0)
        rc  = recall_score(labels_all, preds_all, average=None, zero_division=0)
        metrics[f'f1_{c}'] = f1c[i]
        metrics[f'precision_{c}'] = pc[i]
        metrics[f'recall_{c}'] = rc[i]
    return loss_sum/len(loader.dataset), metrics, labels_all, preds_all

def train_model(config):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    train_loader, val_loader, class_names, wt = get_loaders(config)
    model     = build_model(config)
    criterion = FocalLoss(weight=wt, gamma=config.FOCAL_GAMMA, label_smoothing=config.LABEL_SMOOTHING)
    best_f1, global_ep = 0.0, 0

    for phase, n_epochs, use_aug, use_tta in [
        (1, config.PHASE1_EPOCHS, True,  False),
        (2, config.PHASE2_EPOCHS, True,  False),
        (3, config.PHASE3_EPOCHS, False, True),
    ]:
        print(f"\n{'='*60}\nPHASE {phase}\n{'='*60}")
        if phase > 1:
            ck = torch.load(f'{config.OUTPUT_DIR}/best_model.pth', weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
        unfreeze(model, phase)
        optimizer = make_optimizer(model, config, phase)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-8)
        no_improve = 0

        for ep in range(n_epochs):
            tr_loss, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, config, use_aug)
            vl_loss, vm, _, _ = evaluate(model, val_loader, criterion, config, use_tta)
            scheduler.step()
            vf1 = vm['f1_macro']
            tta_tag = ' (TTA)' if use_tta else ''
            print(f"Ep {global_ep+1:3d} P{phase} | Train F1={tr_f1:.4f} | "
                  f"Val F1={vf1:.4f}{tta_tag} | "
                  f"Blurry F1={vm['f1_Blurry']:.4f} R={vm['recall_Blurry']:.4f}")
            if vf1 > best_f1 + config.MIN_DELTA:
                best_f1 = vf1
                no_improve = 0
                torch.save({'model_state_dict': model.state_dict(), 'val_metrics': vm},
                           f'{config.OUTPUT_DIR}/best_model.pth')
                print(f"  ✓ Best! F1={best_f1:.4f}")
            else:
                no_improve += 1
            global_ep += 1
            if best_f1 >= 0.95: print("🎉 95%+ REACHED!"); break
            if no_improve >= config.PATIENCE: print("Early stop"); break
        if best_f1 >= 0.95: break

    ck = torch.load(f'{config.OUTPUT_DIR}/best_model.pth', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    _, fm, labels, preds = evaluate(model, val_loader, criterion, config, tta=True)
    print(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    print(f"Accuracy: {fm['accuracy']:.4f}  F1 Macro: {fm['f1_macro']:.4f}")
    for c in class_names:
        st = '✓' if fm[f'f1_{c}'] >= 0.90 else '⚠'
        print(f"{st} {c:10s}: F1={fm[f'f1_{c}']:.4f}  P={fm[f'precision_{c}']:.4f}  R={fm[f'recall_{c}']:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=class_names)}")
    with open(f'{config.OUTPUT_DIR}/final_metrics.json','w') as f: json.dump(fm, f, indent=4)
    print(f"✓ Saved to {config.OUTPUT_DIR}")

if __name__ == '__main__':
    train_model(Config())