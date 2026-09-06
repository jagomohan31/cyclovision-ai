"""
Training loop for Model B (classification) — the model the plan says to
get working first, since it's the most tractable of the three.

Run as-is today: it trains on real IMD labels with synthetic placeholder
images (see src/data/dataset.py), so you can confirm the whole loop —
data loading, class weighting, forward/backward pass, per-class F1 — runs
correctly *before* real INSAT imagery is ready. Loss will plateau at a
mediocre level on synthetic images; that's expected and fine. The moment
you drop real .npy INSAT crops into data/raw/insat/<storm_id>/, this same
script starts training on real signal with zero code changes.

Usage:
    python -m src.training.train_classifier --epochs 5
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config import NUM_CATEGORIES, ERA5_FEATURES, CATEGORY_NAMES
from src.data.dataset import CycloneClassificationDataset
from src.models.classification import CycloneClassifier
from src.training.metrics import per_class_f1, confusion_matrix


def compute_class_weights(dataset: CycloneClassificationDataset) -> torch.Tensor:
    """
    Inverse-frequency class weights, fed to CrossEntropyLoss — necessary
    because real cyclone data is heavily imbalanced (2,776 Depressions vs.
    40 Super Cyclonic Storms in the real IMD record; see
    data/raw/ibtracs/SOURCE.md). Without this, the model can hit >85%
    accuracy while never once predicting the rarest, highest-impact classes.
    """
    labels = dataset.obs["category_from_grade"].astype(int).values
    counts = np.bincount(labels, minlength=NUM_CATEGORIES).astype(float)
    counts[counts == 0] = 1  # avoid divide-by-zero for any wholly-absent class
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


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

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_true), np.array(all_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--subset", type=int, default=None,
                         help="Use only the first N samples — for a fast local smoke-test "
                              "on a laptop/CPU before committing to a full Colab/Kaggle GPU run.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = CycloneClassificationDataset()
    if args.subset:
        from torch.utils.data import Subset
        full_dataset = Subset(full_dataset, range(min(args.subset, len(full_dataset))))
        full_dataset.obs = full_dataset.dataset.obs.iloc[full_dataset.indices].reset_index(drop=True)
        print(f"[--subset] Using {len(full_dataset)} samples for a fast smoke-test.")

    # Split by storm ID — not random observation — so no storm's observations
    # appear in both train and val. Random observation splits cause data
    # leakage (the model sees the same storm at t=1 in train and t=2 in val).
    import pandas as pd
    from torch.utils.data import Subset

    base_ds = full_dataset.dataset if hasattr(full_dataset, "dataset") else full_dataset
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
    print(f"Train: {len(train_ds)} obs ({len(train_storms)} storms) | "
          f"Val: {len(val_ds)} obs ({len(val_storms)} storms) — split by storm ID")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = CycloneClassifier(
        num_categories=NUM_CATEGORIES,
        num_era5_features=len(ERA5_FEATURES),
    ).to(device)

    class_weights = compute_class_weights(full_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, y_true, y_pred = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        acc = (y_true == y_pred).mean()
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.3f} "
              f"val_loss={val_loss:.3f} val_acc={acc:.1%}")

    print("\nFinal per-class F1 (validation set):")
    f1s = per_class_f1(y_true, y_pred, num_classes=NUM_CATEGORIES)
    for idx, f1 in f1s.items():
        support = int((y_true == idx).sum())
        if support > 0:
            print(f"  {CATEGORY_NAMES[idx]:<32} F1={f1:.2f}  (support={support})")

    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, num_classes=NUM_CATEGORIES)
    print(cm)

    torch.save(model.state_dict(), "classifier_checkpoint.pt")
    print("\nSaved checkpoint to classifier_checkpoint.pt")


if __name__ == "__main__":
    main()
