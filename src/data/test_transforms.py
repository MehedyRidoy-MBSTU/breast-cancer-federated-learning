from pathlib import Path

from torch.utils.data import DataLoader

from breakhis_dataset import BreaKHisDataset
from transforms import (
    get_train_transforms,
    get_eval_transforms,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = (
    PROJECT_ROOT
    / "processed"
    / "splits"
    / "train.csv"
)

VALIDATION_CSV = (
    PROJECT_ROOT
    / "processed"
    / "splits"
    / "validation.csv"
)


# ============================================================
# TRAIN DATASET
# ============================================================

train_dataset = BreaKHisDataset(
    csv_file=TRAIN_CSV,
    transform=get_train_transforms()
)


# ============================================================
# VALIDATION DATASET
# ============================================================

validation_dataset = BreaKHisDataset(
    csv_file=VALIDATION_CSV,
    transform=get_eval_transforms()
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ============================================================
# TEST TRAIN BATCH
# ============================================================

train_images, train_labels = next(
    iter(train_loader)
)

print("=" * 60)
print("TRAINING BATCH")
print("=" * 60)

print("Images:", train_images.shape)
print("Labels:", train_labels.shape)

print(
    "Pixel range:",
    train_images.min().item(),
    "to",
    train_images.max().item()
)


# ============================================================
# TEST VALIDATION BATCH
# ============================================================

val_images, val_labels = next(
    iter(validation_loader)
)

print("\n" + "=" * 60)
print("VALIDATION BATCH")
print("=" * 60)

print("Images:", val_images.shape)
print("Labels:", val_labels.shape)

print(
    "Pixel range:",
    val_images.min().item(),
    "to",
    val_images.max().item()
)


print("\nTransform test completed successfully!")