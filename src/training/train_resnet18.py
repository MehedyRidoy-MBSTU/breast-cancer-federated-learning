from pathlib import Path
import time
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

from src.models.resnet18 import create_resnet18


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32

NUM_EPOCHS = 15

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

VAL_CSV = (
    PROJECT_ROOT
    / "processed"
    / "splits"
    / "validation.csv"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
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

print("=" * 70)
print("CENTRALIZED RESNET-18 TRAINING")
print("=" * 70)

print("Device:", device)

# ============================================================
# DATASETS
# ============================================================

train_dataset = BreaKHisDataset(
    csv_file=TRAIN_CSV,
    transform=get_train_transforms(IMAGE_SIZE),
)

val_dataset = BreaKHisDataset(
    csv_file=VAL_CSV,
    transform=get_eval_transforms(IMAGE_SIZE),
)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = train_dataset.data["label"]

class_counts = (
    train_labels
    .value_counts()
    .sort_index()
)

print("\nClass counts:")
print(class_counts)

# Weight formula:
#
# weight[class] = total_samples /
#                (number_of_classes * class_samples)

num_classes = 2
total_samples = len(train_labels)

class_weights = []

for class_id in range(num_classes):

    count = class_counts[class_id]

    weight = total_samples / (
        num_classes * count
    )

    class_weights.append(weight)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)

print("\nClass weights:")
print(class_weights)

# ============================================================
# MODEL
# ============================================================

model = create_resnet18(
    num_classes=2,
    pretrained=True,
)

model = model.to(device)

print("\nModel:")
print("ResNet-18")
print("Parameters:",
      sum(p.numel() for p in model.parameters()))

# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)

# ============================================================
# METRICS FUNCTION
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

        auc = 0.0

    # Confusion matrix
    #
    # [[TN, FP],
    #  [FN, TP]]

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
        "auc": auc,
        "confusion_matrix": cm,
    }


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        # Predictions
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities,
    )

    return epoch_loss, metrics


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() *
                images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities,
    )

    return epoch_loss, metrics


# ============================================================
# TRAINING LOOP
# ============================================================

best_f1 = -1.0

best_model_state = None

history = []

training_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):

    epoch_start = time.time()

    train_loss, train_metrics = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
    )

    val_loss, val_metrics = validate(
        model,
        val_loader,
        criterion,
    )

    # Update scheduler
    scheduler.step(
        val_metrics["f1"]
    )

    current_lr = optimizer.param_groups[0]["lr"]

    epoch_time = (
        time.time() -
        epoch_start
    )

    print(
        f"\nEpoch {epoch:02d}/{NUM_EPOCHS}"
    )

    print(
        f"Time: {epoch_time:.1f}s"
    )

    print(
        f"LR: {current_lr:.6f}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Train F1: {train_metrics['f1']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f}"
    )

    print(
        f"Train Acc: {train_metrics['accuracy']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f}"
    )

    print(
        f"Val Precision: {val_metrics['precision']:.4f}"
    )

    print(
        f"Val Recall: {val_metrics['recall']:.4f}"
    )

    print(
        f"Val Specificity: "
        f"{val_metrics['specificity']:.4f}"
    )

    print(
        f"Val ROC-AUC: "
        f"{val_metrics['auc']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_metrics["f1"] > best_f1:

        best_f1 = val_metrics["f1"]

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        checkpoint_file = (
            CHECKPOINT_DIR
            / "resnet18_best.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_f1":
                    val_metrics["f1"],
                "val_auc":
                    val_metrics["auc"],
            },
            checkpoint_file,
        )

        print(
            "✓ Best model saved."
        )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history.append({
        "epoch": epoch,

        "train_loss": train_loss,
        "val_loss": val_loss,

        "train_accuracy":
            train_metrics["accuracy"],

        "val_accuracy":
            val_metrics["accuracy"],

        "train_precision":
            train_metrics["precision"],

        "val_precision":
            val_metrics["precision"],

        "train_recall":
            train_metrics["recall"],

        "val_recall":
            val_metrics["recall"],

        "train_specificity":
            train_metrics["specificity"],

        "val_specificity":
            val_metrics["specificity"],

        "train_f1":
            train_metrics["f1"],

        "val_f1":
            val_metrics["f1"],

        "train_auc":
            train_metrics["auc"],

        "val_auc":
            val_metrics["auc"],

        "learning_rate":
            current_lr,
    })


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_file = (
    RESULTS_DIR
    / "resnet18_training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False
)

# ============================================================
# TRAINING SUMMARY
# ============================================================

total_time = (
    time.time() -
    training_start
)

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Total training time: "
    f"{total_time / 60:.2f} minutes"
)

print(
    f"Best validation F1: "
    f"{best_f1:.4f}"
)

print(
    f"Best checkpoint:\n"
    f"{CHECKPOINT_DIR / 'resnet18_best.pth'}"
)

print(
    f"Training history:\n"
    f"{history_file}"
)

print("=" * 70)