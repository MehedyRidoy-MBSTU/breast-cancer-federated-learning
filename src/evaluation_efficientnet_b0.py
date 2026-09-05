import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.data.breakhis_dataset import BreaKHisDataset
from src.data.transforms import get_eval_transforms
from src.models.efficientnet_b0 import create_efficientnet_b0


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

TEST_CSV = "processed/splits/test.csv"

CHECKPOINT = (
    "checkpoints/efficientnet_b0_best.pth"
)

RESULTS_DIR = "results"

PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR,
    "efficientnet_b0_test_predictions.csv"
)


# ============================================================
# REPRODUCIBILITY
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
print("FINAL TEST EVALUATION - EFFICIENTNET-B0")
print("=" * 70)

print(f"Device: {device}")
print(f"Test CSV: {TEST_CSV}")
print(f"Checkpoint: {CHECKPOINT}")


# ============================================================
# DATASET
# ============================================================

print("\nLoading dataset...")


test_dataset = BreaKHisDataset(
    TEST_CSV,
    transform=get_eval_transforms(IMAGE_SIZE)
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


print(
    f"Test images: {len(test_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")


model = create_efficientnet_b0(
    num_classes=2
)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)


# Support both formats:
# 1. Raw state_dict
# 2. checkpoint dictionary

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(device)

model.eval()


print(
    "Best EfficientNet-B0 checkpoint loaded successfully."
)


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []


total_loss = 0.0


criterion = torch.nn.CrossEntropyLoss()


with torch.no_grad():

    for images, labels in test_loader:


        images = images.to(device)

        labels = labels.to(device)


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        total_loss += (
            loss.item()
            *
            labels.size(0)
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



# ============================================================
# NUMPY CONVERSION
# ============================================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

all_probabilities = np.array(
    all_probabilities
)


test_loss = (
    total_loss
    /
    len(test_dataset)
)



# ============================================================
# METRICS
# ============================================================


accuracy = accuracy_score(
    all_labels,
    all_predictions
)


precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)


recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)


f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)



# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


tn, fp, fn, tp = cm.ravel()


specificity = (
    tn /
    (tn + fp)
)



# ============================================================
# RESULTS
# ============================================================


print("\n" + "=" * 70)
print("FINAL TEST RESULTS - EFFICIENTNET-B0")
print("=" * 70)


print(
    f"Test Loss:       {test_loss:.4f}"
)

print(
    f"Accuracy:        {accuracy:.4f}"
)

print(
    f"Precision:       {precision:.4f}"
)

print(
    f"Recall:          {recall:.4f}"
)

print(
    f"Sensitivity:     {recall:.4f}"
)

print(
    f"Specificity:     {specificity:.4f}"
)

print(
    f"F1-score:        {f1:.4f}"
)

print(
    f"ROC-AUC:         {roc_auc:.4f}"
)



print("\nConfusion Matrix:")

print(cm)



print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Benign",
            "Malignant"
        ],
        digits=4,
        zero_division=0
    )
)



# ============================================================
# SAVE PREDICTIONS
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


prediction_df = (
    test_dataset.data.copy()
)


prediction_df["true_label"] = (
    all_labels
)

prediction_df["predicted_label"] = (
    all_predictions
)

prediction_df["malignant_probability"] = (
    all_probabilities
)



prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)



print("=" * 70)
print(
    "EFFICIENTNET-B0 TEST EVALUATION COMPLETED"
)
print("=" * 70)


print(
    "\nPredictions saved to:"
)

print(
    os.path.abspath(
        PREDICTIONS_FILE
    )
)