import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_FILE = (
    "results/efficientnet_b0_test_predictions.csv"
)

OUTPUT_FILE = (
    "results/efficientnet_b0_threshold_analysis.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EfficientNet-B0 Threshold Analysis")
print("=" * 70)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

df = pd.read_csv(PREDICTIONS_FILE)

y_true = df["true_label"].values
y_prob = df["malignant_probability"].values


print(f"Samples: {len(df)}")


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
    threshold
):

    y_pred = (
        y_prob >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    return {
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }



# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_true,
    y_prob
)



# ============================================================
# THRESHOLD SEARCH
# ============================================================

threshold_results = []


for threshold in np.arange(
    0.01,
    1.00,
    0.01
):

    result = calculate_metrics(
        y_true,
        y_prob,
        threshold
    )

    result["ROC-AUC"] = roc_auc

    threshold_results.append(result)



threshold_df = pd.DataFrame(
    threshold_results
)


best_result = threshold_df.loc[
    threshold_df["F1-score"].idxmax()
]


best_threshold = float(
    best_result["Threshold"]
)



# ============================================================
# DEFAULT 0.50
# ============================================================

default_result = calculate_metrics(
    y_true,
    y_prob,
    0.50
)

default_result["ROC-AUC"] = roc_auc



# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"Threshold: {best_threshold:.2f}"
)

print(
    f"F1-score: {best_result['F1-score']:.4f}"
)

print(
    f"Sensitivity: {best_result['Sensitivity']:.4f}"
)

print(
    f"Specificity: {best_result['Specificity']:.4f}"
)



print("\n" + "=" * 70)
print("DEFAULT THRESHOLD (0.50)")
print("=" * 70)


print(
    f"F1-score: {default_result['F1-score']:.4f}"
)

print(
    f"Sensitivity: {default_result['Sensitivity']:.4f}"
)

print(
    f"Specificity: {default_result['Specificity']:.4f}"
)



# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


save_df = pd.DataFrame(
    [
        {
            "Model": "EfficientNet-B0 Optimized Threshold",
            **best_result.to_dict()
        },
        {
            "Model": "EfficientNet-B0 Default Threshold",
            **default_result
        }
    ]
)


save_df.to_csv(
    OUTPUT_FILE,
    index=False
)



print("\n" + "=" * 70)
print("Threshold analysis completed")
print("=" * 70)

print(
    f"Saved: {OUTPUT_FILE}"
)