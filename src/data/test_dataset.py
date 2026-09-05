from pathlib import Path

import torch
from torch.utils.data import DataLoader

from breakhis_dataset import BreaKHisDataset


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


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)


# ============================================================
# DATASET
# ============================================================

dataset = BreaKHisDataset(
    csv_file=TRAIN_CSV,
    image_size=224
)

print("Dataset:", dataset.csv_file)
print("Number of images:", len(dataset))


# ============================================================
# FIRST IMAGE
# ============================================================

image, label = dataset[0]

print("\nFirst sample:")
print("Image shape:", image.shape)
print("Label:", label.item())
print("Image dtype:", image.dtype)


# ============================================================
# METADATA
# ============================================================

metadata = dataset.get_metadata(0)

print("\nFirst image metadata:")

for key, value in metadata.items():
    print(f"{key}: {value}")


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

images, labels = next(iter(loader))

print("\nFirst batch:")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Labels:", labels)


# ============================================================
# MOVE BATCH TO MPS
# ============================================================

images = images.to(device)
labels = labels.to(device)

print("\nAfter moving to device:")
print("Images device:", images.device)
print("Labels device:", labels.device)

print("\nDataset test completed successfully!")