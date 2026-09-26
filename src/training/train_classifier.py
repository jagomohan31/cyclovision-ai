"""
Training loop for Model B (classification) â€” the model the plan says to
get working first, since it's the most tractable of the three.

Run as-is today: it trains on real IMD labels with synthetic placeholder
images (see src/data/dataset.py), so you can confirm the whole loop â€”
data loading, class weighting, forward/backward pass, per-class F1 â€” runs
correctly *before* real INSAT imagery is ready. Loss will plateau at a
mediocre level on synthetic images; that's expected and fine. The moment
you drop real .npy INSAT crops into data/raw/insat/<storm_id>/, this same
script starts training on real signal with zero code changes.

Regularization (Sep 2026 update):
  - Weight decay (1e-4) on Adam to penalise large weights
  - ReduceLROnPlateau scheduler: halve LR on val-loss plateau
  - Early stopping (patience=3) to save best checkpoint & exit cleanly
  - Augmented training wrapper: random flips on real INSAT frames
  - Dropout added to CNN backbone & ERA5 encoder (see classification.py)

Usage:
    python -m src.training.train_classifier --epochs 20
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import NUM_CATEGORIES, ERA5_FEATURES, CATEGORY_NAMES
from src.data.dataset import CycloneClassificationDataset
from src.models.classification import CycloneClassifier
from src.training.metrics import per_class_f1, confusion_matrix


# ---------------------------------------------------------------------------
# Augmented wrapper dataset â€” applies random flips to training frames only
# ---------------------------------------------------------------------------
class AugmentedSubset(Dataset):
    """Wraps a Subset of CycloneClassificationDataset and enables augmentation."""

    def __init__(self, subset):
        self.subset = subset
        # Resolve the base CycloneClassificationDataset regardless of nesting
        base = subset
        while hasattr(base, "dataset"):
            base = base.dataset
        self._base_ds = base
        self._is_real = hasattr(base, "_mode") and base._mode == "real"

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        real_idx = self.subset.indices[idx]
        if self._is_real:
            return self._base_ds._getitem_real(real_idx, augment=True)
        return self._base_ds._getitem_synthetic(real_idx)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(dataset) -> tuple[torch.Tensor, list[int]]:
    """
    Inverse-frequency class weights, fed to CrossEntropyLoss.
    Also returns the list of *active* class indices (classes that actually
    appear in the data) â€” we zero-weight empty classes so the loss doesn't
    amplify gradient noise on categories like 'Super Cyclonic Storm' that
    have zero training examples.
    """
    base = dataset
    while hasattr(base, "dataset"):
        base = base.dataset
    if hasattr(base, "_mode") and base._mode == "real":
        labels = np.array([r["category"] for r in base._real_records], dtype=int)
    else:
        labels = base.obs["category_from_grade"].astype(int).values
    counts = np.bincount(labels, minlength=NUM_CATEGORIES).astype(float)
    active_classes = [i for i in range(NUM_CATEGORIES) if counts[i] > 0]
    # Zero weight for empty classes (no signal, only noise)
    weights = np.zeros(NUM_CATEGORIES, dtype=np.float32)
    active_counts = counts[active_classes]
    active_weights = active_counts.sum() / (len(active_classes) * active_counts)
    for i, cls in enumerate(active_classes):
        weights[cls] = active_weights[i]
    print(f"Active classes: {[CATEGORY_NAMES[c] for c in active_classes]}")
    for c in active_classes:
        print(f"  Cat {c} ({CATEGORY_NAMES[c][:20]}): {int(counts[c])} frames, weight={weights[c]:.3f}")
    return torch.tensor(weights, dtype=torch.float32), active_classes


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_true, all_pred = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            era5 = batch["era5"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, era5)
            loss = criterion(logits, labels)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_true), np.array(all_pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (epochs without val improvement).")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--backbone", type=str, default=None,
                        help="Pretrained backbone to use: resnet18 or efficientnet_b0. "
                             "Omit for from-scratch SimpleCNN (original behaviour).")
    parser.add_argument("--subset", type=int, default=None,
                         help="Use only the first N samples â€” for a fast local smoke-test "
                              "on a laptop/CPU before committing to a full Colab/Kaggle GPU run.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = CycloneClassificationDataset()
    if args.subset:
        from torch.utils.data import Subset
        full_dataset = Subset(full_dataset, range(min(args.subset, len(full_dataset))))
        print(f"[--subset] Using {len(full_dataset)} samples for a fast smoke-test.")

    from torch.utils.data import Subset

    base_ds = full_dataset
    while hasattr(base_ds, "dataset"):
        base_ds = base_ds.dataset

    if hasattr(base_ds, "_mode") and base_ds._mode == "real":
        # Per-storm temporal split: take the LAST val_split fraction of each
        # storm's frames for val, the rest for train. This guarantees every
        # intensity grade that exists in a storm appears in both train and val
        # proportionally â€” critical when we only have 14 storms total and
        # storm-level splits produce unrepresentative val sets.
        records = base_ds._real_records
        from collections import defaultdict

        # Group record indices by storm, sorted chronologically (npy filenames
        # are YYYYMMDDHHMM so lexicographic = chronological)
        storm_indices: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            storm_indices[r["storm_id"]].append(i)
        # Each storm's indices are already in npy-scan order (chronological)

        train_idx_real, val_idx_real = [], []
        for sid, idxs in storm_indices.items():
            n = len(idxs)
            cut = max(1, int(n * (1 - args.val_split)))
            train_idx_real.extend(idxs[:cut])
            val_idx_real.extend(idxs[cut:])

        raw_train_ds = Subset(full_dataset, train_idx_real)
        val_ds       = Subset(full_dataset, val_idx_real)
        train_ds = AugmentedSubset(raw_train_ds)

        # Show category distribution in val so we can confirm balance
        val_cat_counts = defaultdict(int)
        for i in val_idx_real:
            val_cat_counts[records[i]["category"]] += 1
        val_dist = " | ".join(f"C{c}={n}" for c, n in sorted(val_cat_counts.items()))

        print(f"Dataset Mode: REAL SATELLITE (INSAT-3DR) with augmentation")
        print(f"Train: {len(train_ds)} frames ({len(storm_indices)} storms) | "
              f"Val: {len(val_ds)} frames â€” Per-storm 80/20 temporal split")
        print(f"Val category distribution: {val_dist}")
    else:
        obs = base_ds.obs
        all_storm_ids = np.array(obs["storm_id"].unique(), dtype=str)
        rng_split = np.random.default_rng(42)
        rng_split.shuffle(all_storm_ids)
        n_val_storms = max(1, int(len(all_storm_ids) * args.val_split))
        val_storms = set(all_storm_ids[:n_val_storms])
        train_storms = set(all_storm_ids[n_val_storms:])

        train_idx = obs.index[obs["storm_id"].isin(train_storms)].tolist()
        val_idx   = obs.index[obs["storm_id"].isin(val_storms)].tolist()

        train_ds = Subset(base_ds, train_idx)
        val_ds   = Subset(base_ds, val_idx)
        print(f"Dataset Mode: SYNTHETIC FALLBACK")
        print(f"Train: {len(train_ds)} obs ({len(train_storms)} storms) | "
              f"Val: {len(val_ds)} obs ({len(val_storms)} storms) â€” split by storm ID")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.backbone:
        from src.models.classification import build_pretrained_backbone
        print(f"Using pretrained backbone: {args.backbone}")
        backbone = build_pretrained_backbone(
            name=args.backbone,
            embed_dim=128,
            in_channels=1,
        )
        model = CycloneClassifier(
            num_categories=NUM_CATEGORIES,
            num_era5_features=len(ERA5_FEATURES),
            image_backbone=backbone,
        ).to(device)
    else:
        model = CycloneClassifier(
            num_categories=NUM_CATEGORIES,
            num_era5_features=len(ERA5_FEATURES),
        ).to(device)

    class_weights, active_classes = compute_class_weights(full_dataset)
    class_weights = class_weights.to(device)

    # Focal loss: down-weights easy majority-class predictions harder than
    # plain weighted cross-entropy. gamma=2 is standard; combined with class
    # weights it strongly forces the model to attend to Depression/CS/ESCS.
    class FocalLoss(nn.Module):
        def __init__(self, weight, gamma=2.0, label_smoothing=0.1):
            super().__init__()
            self.weight = weight
            self.gamma = gamma
            self.label_smoothing = label_smoothing

        def forward(self, logits, targets):
            ce = nn.functional.cross_entropy(
                logits, targets, weight=self.weight,
                label_smoothing=self.label_smoothing, reduction="none"
            )
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.gamma) * ce
            return focal.mean()

    criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.1)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Backup existing checkpoint before retraining
    import shutil, time
    from pathlib import Path
    ckpt_path = Path("classifier_checkpoint.pt")
    if ckpt_path.exists() and not Path("classifier_checkpoint_old.pt").exists():
        shutil.copy(ckpt_path, "classifier_checkpoint_old.pt")
        print("Backed up existing weights to classifier_checkpoint_old.pt")

    best_val_loss = float("inf")
    no_improve    = 0
    y_true_best   = np.array([])
    y_pred_best   = np.array([])

    print(f"\nTraining with: lr={args.lr}, weight_decay={args.weight_decay}, "
          f"patience={args.patience}, epochs={args.epochs}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, _, _         = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   y_true, y_pred = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        dt = time.time() - t0

        scheduler.step(val_loss)

        acc = (y_true == y_pred).mean()
        saved_marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), "classifier_checkpoint.pt")
            saved_marker = " * [Saved Best Checkpoint]"
            y_true_best  = y_true.copy()
            y_pred_best  = y_pred.copy()
        else:
            no_improve += 1

        print(f"Epoch {epoch:02d}/{args.epochs:02d} ({dt:.1f}s) | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={acc:.1%}{saved_marker}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping triggered (no improvement for {args.patience} epochs).")
            break

    print("\nFinal per-class F1 (best validation checkpoint):")
    f1s = per_class_f1(y_true_best, y_pred_best, num_classes=NUM_CATEGORIES)
    for idx, f1 in f1s.items():
        support = int((y_true_best == idx).sum())
        if support > 0:
            print(f"  {CATEGORY_NAMES[idx]:<32} F1={f1:.2f}  (support={support})")

    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true_best, y_pred_best, num_classes=NUM_CATEGORIES)
    print(cm)
    print("\nTraining completed. Best weights saved to classifier_checkpoint.pt")


if __name__ == "__main__":
    main()

