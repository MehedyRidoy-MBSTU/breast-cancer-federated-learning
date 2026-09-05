import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_FILE = (
    "results/densenet121_test_predictions.csv"
)

OUTPUT_FILE = (
    "results/densenet121_threshold_analysis.csv"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("DenseNet-121 Threshold Analysis")
print("=" * 70)


df = pd.read_csv(
    PREDICTION_FILE
)


y_true = df["true_label"].values

y_prob = df[
    "malignant_probability"
].values


print(f"Samples: {len(df)}")


# ============================================================
# THRESHOLD SEARCH
# ============================================================

thresholds = np.arange(
    0.01,
    1.00,
    0.01
)


results = []


for threshold in thresholds:

    y_pred = (
        y_prob >= threshold
    ).astype(int)


    cm = confusion_matrix(
        y_true,
        y_pred
    )


    tn, fp, fn, tp = cm.ravel()


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

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    results.append({

        "threshold": threshold,

        "accuracy": accuracy,

        "precision": precision,

        "sensitivity": recall,

        "specificity": specificity,

        "f1_score": f1,

        "tn": tn,

        "fp": fp,

        "fn": fn,

        "tp": tp
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

best_row = results_df.loc[
    results_df["f1_score"].idxmax()
]


print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)


print(
    f"Threshold: "
    f"{best_row['threshold']:.2f}"
)

print(
    f"F1-score: "
    f"{best_row['f1_score']:.4f}"
)

print(
    f"Sensitivity: "
    f"{best_row['sensitivity']:.4f}"
)

print(
    f"Specificity: "
    f"{best_row['specificity']:.4f}"
)


# ============================================================
# DEFAULT THRESHOLD 0.50
# ============================================================

default_row = results_df[
    results_df["threshold"] == 0.50
].iloc[0]


print("\n" + "=" * 70)
print("DEFAULT THRESHOLD (0.50)")
print("=" * 70)


print(
    f"F1-score: "
    f"{default_row['f1_score']:.4f}"
)

print(
    f"Sensitivity: "
    f"{default_row['sensitivity']:.4f}"
)

print(
    f"Specificity: "
    f"{default_row['specificity']:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("Threshold analysis completed")
print("=" * 70)

print(
    f"Saved: {OUTPUT_FILE}"
)