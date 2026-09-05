from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.data.breakhis_dataset import BreaKHisDataset
from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
)
from src.models.densenet121 import create_densenet121


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = PROJECT_ROOT / "processed" / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "processed" / "splits" / "validation.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32

EPOCHS = 15

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PATIENCE = 2


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print("=" * 70)
print("DenseNet-121 Centralized Training")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")


# ============================================================
# Dataset
# ============================================================

print("\nLoading datasets...")


train_dataset = BreaKHisDataset(
    csv_file=TRAIN_CSV,
    transform=get_train_transforms()
)

val_dataset = BreaKHisDataset(
    csv_file=VAL_CSV,
    transform=get_eval_transforms()
)


print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)


# ============================================================
# Class weights
# ============================================================

train_labels = train_dataset.data["label"].astype(int)

class_counts = np.bincount(
    train_labels,
    minlength=2
)

total_samples = class_counts.sum()

class_weights = total_samples / (
    2 * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)


print("\nClass distribution:")

print(
    f"Benign (0): {class_counts[0]}"
)

print(
    f"Malignant (1): {class_counts[1]}"
)

print("\nClass weights:")

print(
    f"Benign: {class_weights[0].item():.4f}"
)

print(
    f"Malignant: {class_weights[1].item():.4f}"
)


# ============================================================
# Model
# ============================================================

print("\nCreating DenseNet-121...")

model = create_densenet121(
    num_classes=2
)

model = model.to(DEVICE)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Total parameters: {num_parameters:,}"
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Scheduler
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=PATIENCE,
)


# ============================================================
# Metric calculation
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:
        auc = roc_auc_score(
            y_true,
            y_prob
        )
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


# ============================================================
# Training history
# ============================================================

history = []

best_val_f1 = -float("inf")
best_epoch = 0


# ============================================================
# Training loop
# ============================================================

for epoch in range(1, EPOCHS + 1):

    print("\n" + "=" * 70)
    print(f"Epoch {epoch}/{EPOCHS}")
    print("=" * 70)


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    train_true = []
    train_pred = []
    train_prob = []


    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).long()

        train_true.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        train_pred.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        train_prob.extend(
            probabilities.detach()
            .cpu()
            .numpy()
        )


    train_loss = (
        running_loss /
        len(train_dataset)
    )


    train_metrics = calculate_metrics(
        train_true,
        train_pred,
        train_prob
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0

    val_true = []
    val_pred = []
    val_prob = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            predictions = (
                probabilities >= 0.5
            ).long()

            val_true.extend(
                labels.cpu().numpy()
            )

            val_pred.extend(
                predictions.cpu().numpy()
            )

            val_prob.extend(
                probabilities.cpu().numpy()
            )


    val_loss = (
        val_running_loss /
        len(val_dataset)
    )


    val_metrics = calculate_metrics(
        val_true,
        val_pred,
        val_prob
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_metrics["f1"]
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nTrain Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"Train Precision: "
        f"{train_metrics['precision']:.4f}"
    )

    print(
        f"Train Recall: "
        f"{train_metrics['recall']:.4f}"
    )

    print(
        f"Train Specificity: "
        f"{train_metrics['specificity']:.4f}"
    )

    print(
        f"Train F1: "
        f"{train_metrics['f1']:.4f}"
    )

    print(
        f"Train ROC-AUC: "
        f"{train_metrics['roc_auc']:.4f}"
    )


    print(
        f"\nValidation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"Validation Precision: "
        f"{val_metrics['precision']:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{val_metrics['recall']:.4f}"
    )

    print(
        f"Validation Specificity: "
        f"{val_metrics['specificity']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{val_metrics['f1']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{val_metrics['roc_auc']:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.8f}"
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history.append({
        "epoch": epoch,

        "train_loss": train_loss,
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_specificity": train_metrics["specificity"],
        "train_f1": train_metrics["f1"],
        "train_roc_auc": train_metrics["roc_auc"],

        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_specificity": val_metrics["specificity"],
        "val_f1": val_metrics["f1"],
        "val_roc_auc": val_metrics["roc_auc"],

        "learning_rate": current_lr,
    })


    # --------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------

    if val_metrics["f1"] > best_val_f1:

        best_val_f1 = val_metrics["f1"]
        best_epoch = epoch

        checkpoint_path = (
            CHECKPOINT_DIR /
            "densenet121_best.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "seed": SEED,
            },
            checkpoint_path
        )

        print(
            f"\n*** New best model! "
            f"Validation F1 = "
            f"{best_val_f1:.4f}"
        )


# ============================================================
# Save history
# ============================================================

history_df = pd.DataFrame(history)

history_path = (
    RESULTS_DIR /
    "densenet121_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ============================================================
# Final message
# ============================================================

print("\n" + "=" * 70)
print("DenseNet-121 training completed.")
print("=" * 70)

print(
    f"Best validation F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Checkpoint: "
    f"{CHECKPOINT_DIR / 'densenet121_best.pth'}"
)

print(
    f"History: "
    f"{history_path}"
)