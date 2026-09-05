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
    balanced_accuracy_score,
    confusion_matrix,
)

from src.data.breakhis_dataset import BreaKHisDataset
from src.data.transforms import get_eval_transforms
from src.models.resnet18 import create_resnet18


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

VALIDATION_CSV = "processed/splits/validation.csv"
TEST_CSV = "processed/splits/test.csv"
CHECKPOINT = "checkpoints/resnet18_best.pth"

RESULTS_DIR = "results"

VALIDATION_PREDICTIONS = os.path.join(
    RESULTS_DIR,
    "resnet18_validation_predictions.csv"
)

THRESHOLD_RESULTS = os.path.join(
    RESULTS_DIR,
    "resnet18_threshold_analysis.csv"
)

TEST_THRESHOLD_RESULTS = os.path.join(
    RESULTS_DIR,
    "resnet18_test_threshold_evaluation.csv"
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
# LOAD MODEL
# ============================================================

print("=" * 70)
print("RESNET-18 THRESHOLD ANALYSIS")
print("=" * 70)

print(f"Device: {device}")

model = create_resnet18(
    num_classes=2,
    pretrained=False
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Best ResNet-18 checkpoint loaded successfully.")


# ============================================================
# FUNCTION: GET PREDICTIONS
# ============================================================

def get_predictions(csv_file):

    dataset = BreaKHisDataset(
        csv_file,
        transform=get_eval_transforms(IMAGE_SIZE)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    return dataset, np.array(all_labels), np.array(all_probabilities)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 70)

val_dataset, val_labels, val_probabilities = get_predictions(
    VALIDATION_CSV
)

print(f"Validation images: {len(val_labels)}")

val_auc = roc_auc_score(
    val_labels,
    val_probabilities
)

print(f"Validation ROC-AUC: {val_auc:.4f}")


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

val_prediction_df = val_dataset.data.copy()

val_prediction_df["true_label"] = val_labels
val_prediction_df["malignant_probability"] = val_probabilities

val_prediction_df.to_csv(
    VALIDATION_PREDICTIONS,
    index=False
)

print("\nValidation predictions saved to:")
print(os.path.abspath(VALIDATION_PREDICTIONS))


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

results = []

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        val_labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    accuracy = accuracy_score(
        val_labels,
        predictions
    )

    precision = precision_score(
        val_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        val_labels,
        predictions,
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        val_labels,
        predictions
    )

    results.append({
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })


threshold_df = pd.DataFrame(results)

threshold_df.to_csv(
    THRESHOLD_RESULTS,
    index=False
)


# ============================================================
# BEST THRESHOLDS
# ============================================================

best_f1 = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_balanced = threshold_df.loc[
    threshold_df["balanced_accuracy"].idxmax()
]

print("\nBest threshold according to F1:")
print(
    best_f1.to_string()
)

print("\nBest threshold according to Balanced Accuracy:")
print(
    best_balanced.to_string()
)


# ============================================================
# SELECT THRESHOLD
# ============================================================

# For this thesis, use validation F1 to select
# the operating threshold.

selected_threshold = float(
    best_f1["threshold"]
)

print("\n" + "=" * 70)
print("SELECTED VALIDATION THRESHOLD")
print("=" * 70)

print(
    f"Selected threshold: {selected_threshold:.2f}"
)

print(
    "Selection criterion: Maximum validation F1-score"
)


# ============================================================
# TEST EVALUATION USING SELECTED THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("APPLYING SELECTED THRESHOLD TO TEST SET")
print("=" * 70)

test_dataset, test_labels, test_probabilities = get_predictions(
    TEST_CSV
)

print(f"Test images: {len(test_labels)}")

# IMPORTANT:
# The threshold was selected using VALIDATION data.
# The TEST set is only evaluated here.

test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_sensitivity = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_balanced_accuracy = balanced_accuracy_score(
    test_labels,
    test_predictions
)

test_auc = roc_auc_score(
    test_labels,
    test_probabilities
)

tn, fp, fn, tp = confusion_matrix(
    test_labels,
    test_predictions,
    labels=[0, 1]
).ravel()

test_specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS USING VALIDATION-SELECTED THRESHOLD")
print("=" * 70)

print(f"Threshold:           {selected_threshold:.2f}")
print(f"Accuracy:            {test_accuracy:.4f}")
print(f"Precision:           {test_precision:.4f}")
print(f"Sensitivity:         {test_sensitivity:.4f}")
print(f"Specificity:         {test_specificity:.4f}")
print(f"F1-score:            {test_f1:.4f}")
print(f"Balanced Accuracy:   {test_balanced_accuracy:.4f}")
print(f"ROC-AUC:             {test_auc:.4f}")


print("\nConfusion Matrix:")
print(
    np.array([
        [tn, fp],
        [fn, tp]
    ])
)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

test_result_df = pd.DataFrame([{
    "threshold": selected_threshold,
    "accuracy": test_accuracy,
    "precision": test_precision,
    "sensitivity": test_sensitivity,
    "specificity": test_specificity,
    "f1": test_f1,
    "balanced_accuracy": test_balanced_accuracy,
    "roc_auc": test_auc,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp
}])

test_result_df.to_csv(
    TEST_THRESHOLD_RESULTS,
    index=False
)

print("\nTest threshold results saved to:")
print(os.path.abspath(TEST_THRESHOLD_RESULTS))


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 70)