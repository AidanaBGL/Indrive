import os
import json
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, datasets, models
from torchvision.models import VGG16_Weights

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


# ---------------------- Utils ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_dataloaders(data_dir, img_size=224, batch_size=32, num_workers=4,
                     val_size=0.15, test_size=0.10, seed=42):
    """
    Ожидает data_dir с 2 подпапками-классами (например, 'dirty cars' и 'clean cars').
    Делает стратифицированный train/val/test сплит.
    """
    set_seed(seed)

    # Трансформации
    mean, std = VGG16_Weights.IMAGENET1K_V1.transforms().mean, VGG16_Weights.IMAGENET1K_V1.transforms().std
    train_tfms = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Целый датасет
    full_ds = datasets.ImageFolder(root=data_dir, transform=train_tfms)  # временно train_tfms (заменим ниже для val/test)
    targets = np.array(full_ds.targets)

    # Индексы для стратифицированного сплита
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(len(targets)), targets))
    y_trainval = targets[trainval_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(sss2.split(np.zeros(len(trainval_idx)), y_trainval))

    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    # Поднаборы
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(datasets.ImageFolder(root=data_dir, transform=eval_tfms), val_idx)   # с eval_tfms
    test_ds = Subset(datasets.ImageFolder(root=data_dir, transform=eval_tfms), test_idx)

    # Balanced sampler (на случай дисбаланса)
    class_sample_count = np.bincount(targets[train_idx])
    class_weights = 1. / np.maximum(class_sample_count, 1)
    sample_weights = class_weights[targets[train_idx]]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                    num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    class_names = full_ds.classes  # порядок классов
    return train_loader, val_loader, test_loader, class_names


def build_model(num_classes=2, pretrained=True):
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)
    in_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_feats, num_classes)  # под CE loss
    return model


def train_one_epoch(model, loader, device, criterion, optimizer, scaler=None):
    model.train()
    epoch_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return epoch_loss / len(loader.dataset), acc, f1


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)

        all_logits.append(logits.softmax(dim=1).detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    probs = np.concatenate(all_logits)  # shape [N,2]
    y_true = np.concatenate(all_targets)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, probs[:, 1])
    except ValueError:
        auroc = float("nan")

    return total_loss / len(loader.dataset), acc, f1, auroc, y_pred, y_true


def plot_confusion(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True', xlabel='Pred')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Папка-родитель, внутри которой две подпапки классов (dirty/clean).")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--frozen_epochs", type=int, default=3, help="Сколько эпох держать VGG16 замороженной.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="runs/vgg16_binary")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names = make_dataloaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, val_size=args.val_size,
        test_size=args.test_size, seed=args.seed
    )
    print("Classes:", class_names)

    model = build_model(num_classes=2, pretrained=True)
    model.to(device)

    # Заморозка фич-экстрактора на несколько эпох
    for p in model.features.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_f1 = -1.0
    patience, no_improve = 5, 0
    best_path = os.path.join(args.out_dir, "vgg16_dirty_clean_best.pth")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Разморозить после frozen_epochs
        if epoch == args.frozen_epochs + 1:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.frozen_epochs)
            print("Unfroze VGG16 feature extractor.")

        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_loss, val_acc, val_f1, val_auroc, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"[{epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} f1={train_f1:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f} auroc={val_auroc:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_names": class_names,
                "img_size": args.img_size
            }, best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    print(f"Training done in {(time.time()-t0)/60:.1f} min. Best val F1={best_val_f1:.3f}")

    # ---- Тест ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, test_f1, test_auroc, y_pred, y_true = evaluate(model, test_loader, device, criterion)
    print("\nTEST:")
    print(f"loss={test_loss:.4f} acc={test_acc:.3f} f1={test_f1:.3f} auroc={test_auroc:.3f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plot_confusion(cm, class_names, cm_path)
    print(f"Saved confusion matrix to: {cm_path}")

    # Сохранить метаданные
    meta = {
        "classes": class_names,
        "img_size": args.img_size,
        "best_checkpoint": best_path,
        "test_metrics": {
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "f1": float(test_f1),
            "auroc": float(test_auroc)
        }
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved meta to {os.path.join(args.out_dir, 'meta.json')}")


if __name__ == "__main__":
    main()
