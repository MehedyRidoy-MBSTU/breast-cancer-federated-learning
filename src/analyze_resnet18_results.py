import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_PREDICTIONS = (
    "results/resnet18_test_predictions.csv"
)

OUTPUT_DIR = "results/resnet18_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD TEST PREDICTIONS
# ============================================================

print("=" * 70)
print("RESNET-18 TEST RESULT ANALYSIS")
print("=" * 70)

df = pd.read_csv(TEST_PREDICTIONS)

print(f"Test samples: {len(df)}")

true_labels = df["true_label"].values
probabilities = df["malignant_probability"].values


# ============================================================
# ROC-AUC
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    true_labels,
    probabilities
)

roc_auc = auc(
    fpr,
    tpr
)


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        true_labels,
        probabilities
    )
)

pr_auc = average_precision_score(
    true_labels,
    probabilities
)


# ============================================================
# YOUden's J
# ============================================================

youden_j = tpr - fpr

best_youden_index = np.argmax(
    youden_j
)

youden_threshold = roc_thresholds[
    best_youden_index
]

youden_sensitivity = tpr[
    best_youden_index
]

youden_specificity = 1 - fpr[
    best_youden_index
]


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = np.arange(
    0.01,
    1.00,
    0.01
)

threshold_results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        predictions
    )

    threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# BEST F1 ON TEST
# NOTE:
# This is descriptive only.
# We MUST NOT use this threshold for model selection.
# ============================================================

best_f1_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]


# ============================================================
# SENSITIVITY-CONSTRAINED THRESHOLDS
# ============================================================

target_sensitivities = [
    0.90,
    0.95
]

sensitivity_rows = []

for target in target_sensitivities:

    eligible = threshold_df[
        threshold_df["sensitivity"] >= target
    ]

    if len(eligible) > 0:

        # Among thresholds satisfying the
        # sensitivity requirement, choose
        # the one with highest specificity.

        best = eligible.loc[
            eligible["specificity"].idxmax()
        ]

        sensitivity_rows.append({
            "target_sensitivity": target,
            "threshold": best["threshold"],
            "actual_sensitivity": best["sensitivity"],
            "specificity": best["specificity"],
            "precision": best["precision"],
            "f1": best["f1"],
            "accuracy": best["accuracy"]
        })


sensitivity_df = pd.DataFrame(
    sensitivity_rows
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DISCRIMINATION METRICS")
print("=" * 70)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")


print("\n" + "=" * 70)
print("YOUDEN'S J")
print("=" * 70)

print(f"Threshold:   {youden_threshold:.4f}")
print(f"Sensitivity: {youden_sensitivity:.4f}")
print(f"Specificity: {youden_specificity:.4f}")
print(f"Youden's J:  {youden_j[best_youden_index]:.4f}")


print("\n" + "=" * 70)
print("DESCRIPTIVE TEST-SET F1 OPTIMUM")
print("=" * 70)

print(
    f"Threshold:   {best_f1_row['threshold']:.2f}"
)

print(
    f"Accuracy:    {best_f1_row['accuracy']:.4f}"
)

print(
    f"Precision:   {best_f1_row['precision']:.4f}"
)

print(
    f"Sensitivity: {best_f1_row['sensitivity']:.4f}"
)

print(
    f"Specificity: {best_f1_row['specificity']:.4f}"
)

print(
    f"F1:          {best_f1_row['f1']:.4f}"
)

print(
    "\nIMPORTANT: This test-set threshold is "
    "DESCRIPTIVE ONLY and must NOT be used "
    "for model selection."
)


print("\n" + "=" * 70)
print("SENSITIVITY-CONSTRAINED ANALYSIS")
print("=" * 70)

if len(sensitivity_df) > 0:
    print(
        sensitivity_df.to_string(
            index=False
        )
    )
else:
    print(
        "No threshold satisfied the requested "
        "sensitivity constraints."
    )


# ============================================================
# SAVE TABLES
# ============================================================

threshold_file = os.path.join(
    OUTPUT_DIR,
    "threshold_analysis_test.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False
)

sensitivity_file = os.path.join(
    OUTPUT_DIR,
    "sensitivity_constrained_thresholds.csv"
)

sensitivity_df.to_csv(
    sensitivity_file,
    index=False
)


# ============================================================
# SAVE ROC DATA
# ============================================================

roc_df = pd.DataFrame({
    "false_positive_rate": fpr,
    "true_positive_rate": tpr,
    "threshold": roc_thresholds
})

roc_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "roc_curve_data.csv"
    ),
    index=False
)


# ============================================================
# SAVE PR DATA
# ============================================================

pr_df = pd.DataFrame({
    "precision": precision_curve,
    "recall": recall_curve
})

pr_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "precision_recall_curve_data.csv"
    ),
    index=False
)


# ============================================================
# FIGURE 1 — ROC CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"ResNet-18 (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel(
    "False Positive Rate (1 - Specificity)"
)

plt.ylabel(
    "True Positive Rate (Sensitivity)"
)

plt.title(
    "ResNet-18 ROC Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

roc_file = os.path.join(
    OUTPUT_DIR,
    "resnet18_roc_curve.png"
)

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2 — PRECISION-RECALL CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_curve,
    precision_curve,
    label=f"ResNet-18 (AP = {pr_auc:.4f})"
)

plt.xlabel(
    "Recall / Sensitivity"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "ResNet-18 Precision-Recall Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

pr_file = os.path.join(
    OUTPUT_DIR,
    "resnet18_precision_recall_curve.png"
)

plt.savefig(
    pr_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3 — THRESHOLD PERFORMANCE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["sensitivity"],
    label="Sensitivity"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["specificity"],
    label="Specificity"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1"],
    label="F1-score"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    label="Precision"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Metric"
)

plt.title(
    "ResNet-18 Performance Across Classification Thresholds"
)

plt.ylim(
    0,
    1.05
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

threshold_plot = os.path.join(
    OUTPUT_DIR,
    "resnet18_threshold_performance.png"
)

plt.savefig(
    threshold_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4 — MALIGNANT PROBABILITY DISTRIBUTION
# ============================================================

benign_probabilities = probabilities[
    true_labels == 0
]

malignant_probabilities = probabilities[
    true_labels == 1
]

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    benign_probabilities,
    bins=30,
    alpha=0.6,
    label="Benign"
)

plt.hist(
    malignant_probabilities,
    bins=30,
    alpha=0.6,
    label="Malignant"
)

plt.axvline(
    0.50,
    linestyle="--",
    label="Threshold = 0.50"
)

plt.axvline(
    0.10,
    linestyle=":",
    label="Threshold = 0.10"
)

plt.xlabel(
    "Predicted Probability of Malignancy"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "ResNet-18 Predicted Probability Distribution"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

distribution_file = os.path.join(
    OUTPUT_DIR,
    "resnet18_probability_distribution.png"
)

plt.savefig(
    distribution_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CONFUSION MATRICES
# ============================================================

def save_confusion_matrix(
    labels,
    probabilities,
    threshold,
    filename,
    title
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(2)

    plt.xticks(
        tick_marks,
        ["Benign", "Malignant"]
    )

    plt.yticks(
        tick_marks,
        ["Benign", "Malignant"]
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


save_confusion_matrix(
    true_labels,
    probabilities,
    0.50,
    "resnet18_confusion_matrix_050.png",
    "ResNet-18 Confusion Matrix — Threshold 0.50"
)

save_confusion_matrix(
    true_labels,
    probabilities,
    0.10,
    "resnet18_confusion_matrix_010.png",
    "ResNet-18 Confusion Matrix — Threshold 0.10"
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "ROC_AUC": roc_auc,
    "PR_AUC": pr_auc,
    "Youden_threshold": youden_threshold,
    "Youden_sensitivity": youden_sensitivity,
    "Youden_specificity": youden_specificity,
    "Test_F1_optimal_threshold_descriptive": best_f1_row["threshold"],
    "Test_F1_optimal_descriptive": best_f1_row["f1"]
}])

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "resnet18_analysis_summary.csv"
    ),
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print("\nFiles saved in:")
print(os.path.abspath(OUTPUT_DIR))

print("\nGenerated figures:")
print("1. resnet18_roc_curve.png")
print("2. resnet18_precision_recall_curve.png")
print("3. resnet18_threshold_performance.png")
print("4. resnet18_probability_distribution.png")
print("5. resnet18_confusion_matrix_050.png")
print("6. resnet18_confusion_matrix_010.png")