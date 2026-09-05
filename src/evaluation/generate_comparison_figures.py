import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "results/comparison"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


FEDAVG_PRED = (
    "results/fedavg_densenet121_predictions.csv"
)

FEDPROX_PRED = (
    "results/fedprox_densenet121_predictions.csv"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

fedavg = pd.read_csv(
    FEDAVG_PRED
)

fedprox = pd.read_csv(
    FEDPROX_PRED
)


# ============================================================
# ROC CURVE
# ============================================================

plt.figure(
    figsize=(7,6)
)


for df, name in [
    (fedavg, "FedAvg DenseNet-121"),
    (fedprox, "FedProx DenseNet-121")
]:

    fpr, tpr, _ = roc_curve(
        df["true_label"],
        df["probability"]
    )

    roc_auc = auc(
        fpr,
        tpr
    )


    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={roc_auc:.3f})"
    )


plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve Comparison"
)


plt.legend()

plt.grid()

plt.tight_layout()


plt.savefig(
    f"{OUTPUT_DIR}/roc_curve_comparison.png",
    dpi=300
)


plt.close()



# ============================================================
# CONFUSION MATRICES
# ============================================================


def save_confusion_matrix(
    y_true,
    y_pred,
    title,
    filename
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )


    plt.figure(
        figsize=(5,4)
    )


    plt.imshow(cm)


    plt.title(
        title
    )


    plt.colorbar()


    classes = [
        "Benign",
        "Malignant"
    ]


    plt.xticks(
        [0,1],
        classes
    )


    plt.yticks(
        [0,1],
        classes
    )


    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i,j],
                ha="center",
                va="center"
            )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )


    plt.tight_layout()


    plt.savefig(
        filename,
        dpi=300
    )


    plt.close()



save_confusion_matrix(
    fedavg["true_label"],
    fedavg["prediction"],
    "FedAvg DenseNet-121 Confusion Matrix",
    f"{OUTPUT_DIR}/confusion_matrix_fedavg.png"
)



save_confusion_matrix(
    fedprox["true_label"],
    fedprox["prediction"],
    "FedProx DenseNet-121 Confusion Matrix",
    f"{OUTPUT_DIR}/confusion_matrix_fedprox.png"
)



print("="*70)
print("COMPARISON FIGURES GENERATED")
print("="*70)


print(
    "Saved inside:",
    OUTPUT_DIR
)