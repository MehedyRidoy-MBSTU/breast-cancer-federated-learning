import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.data.breakhis_dataset import BreaKHisDataset
from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms
)

from src.models.efficientnet_b0 import create_efficientnet_b0


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0


TRAIN_CSV = "processed/splits/train.csv"
VAL_CSV = "processed/splits/validation.csv"

CHECKPOINT_DIR = "checkpoints"

RESULTS_DIR = "results"

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "efficientnet_b0_best.pth"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "efficientnet_b0_training_history.csv"
)


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EfficientNet-B0 Centralized Training")
print("=" * 70)

print(f"Device: {device}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")


# ============================================================
# DATASET
# ============================================================

print("\nLoading datasets...")


train_dataset = BreaKHisDataset(
    TRAIN_CSV,
    transform=get_train_transforms(IMAGE_SIZE)
)


val_dataset = BreaKHisDataset(
    VAL_CSV,
    transform=get_eval_transforms(IMAGE_SIZE)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = train_dataset.data["label"].values


class_counts = np.bincount(train_labels)


class_weights = torch.tensor(
    [
        len(train_labels) / (2 * class_counts[0]),
        len(train_labels) / (2 * class_counts[1])
    ],
    dtype=torch.float32
)


class_weights = class_weights.to(device)


print("\nClass distribution:")
print(f"Benign (0): {class_counts[0]}")
print(f"Malignant (1): {class_counts[1]}")


print("\nClass weights:")
print(f"Benign: {class_weights[0].item():.4f}")
print(f"Malignant: {class_weights[1].item():.4f}")


# ============================================================
# MODEL
# ============================================================

print("\nCreating EfficientNet-B0...")


model = create_efficientnet_b0(
    num_classes=2
)


model = model.to(device)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


print(
    f"Total parameters: {total_parameters:,}"
)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# TRAINING
# ============================================================

history = []

best_f1 = 0.0
best_epoch = 0


for epoch in range(EPOCHS):

    print("\n" + "=" * 70)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("=" * 70)


    # -------------------------
    # TRAIN
    # -------------------------

    model.train()


    train_loss = 0

    train_labels = []
    train_predictions = []
    train_probabilities = []


    for images, labels in train_loader:


        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        train_loss += (
            loss.item()
            *
            labels.size(0)
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )[:,1]


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_labels.extend(
            labels.cpu().numpy()
        )

        train_predictions.extend(
            predictions.cpu().numpy()
        )

        train_probabilities.extend(
            probabilities.detach()
            .cpu()
            .numpy()
        )


    train_loss /= len(train_dataset)


    train_metrics = calculate_metrics(
        np.array(train_labels),
        np.array(train_predictions),
        np.array(train_probabilities)
    )


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()


    val_loss = 0

    val_labels = []
    val_predictions = []
    val_probabilities = []


    with torch.no_grad():

        for images, labels in val_loader:


            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_loss += (
                loss.item()
                *
                labels.size(0)
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:,1]


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_labels.extend(
                labels.cpu().numpy()
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_probabilities.extend(
                probabilities.cpu().numpy()
            )


    val_loss /= len(val_dataset)


    val_metrics = calculate_metrics(
        np.array(val_labels),
        np.array(val_predictions),
        np.array(val_probabilities)
    )


    scheduler.step(
        val_metrics["f1"]
    )


    current_lr = optimizer.param_groups[0]["lr"]


    print("\nTrain Loss:", round(train_loss,4))

    for key,value in train_metrics.items():
        print(
            f"Train {key.capitalize()}: {value:.4f}"
        )


    print("\nValidation Loss:", round(val_loss,4))

    for key,value in val_metrics.items():
        print(
            f"Validation {key.capitalize()}: {value:.4f}"
        )


    print(
        f"Learning Rate: {current_lr:.8f}"
    )


    history.append({

        "epoch": epoch+1,

        "train_loss": train_loss,

        **{
            f"train_{k}":v
            for k,v in train_metrics.items()
        },


        "val_loss": val_loss,

        **{
            f"val_{k}":v
            for k,v in val_metrics.items()
        },


        "learning_rate": current_lr

    })


    # SAVE BEST MODEL

    if val_metrics["f1"] > best_f1:

        best_f1 = val_metrics["f1"]

        best_epoch = epoch + 1


        os.makedirs(
            CHECKPOINT_DIR,
            exist_ok=True
        )


        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )


        print(
            f"\n*** New best model! Validation F1 = {best_f1:.4f}"
        )


# ============================================================
# SAVE HISTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


history_df = pd.DataFrame(history)


history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# FINISH
# ============================================================


print("\n" + "=" * 70)
print("EfficientNet-B0 training completed.")
print("=" * 70)

print(
    f"Best validation F1: {best_f1:.4f}"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Checkpoint: {CHECKPOINT_PATH}"
)

print(
    f"History: {HISTORY_PATH}"
)