import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

FEDAVG_RESULTS = (
    "results/fedavg_densenet121_test_results.csv"
)

FEDPROX_RESULTS = (
    "results/fedprox_densenet121_test_results.csv"
)


OUTPUT_DIR = (
    "results/comparison"
)


CSV_OUTPUT = (
    f"{OUTPUT_DIR}/model_comparison.csv"
)


BAR_OUTPUT = (
    f"{OUTPUT_DIR}/performance_comparison.png"
)


ROC_OUTPUT = (
    f"{OUTPUT_DIR}/roc_auc_comparison.png"
)


RADAR_OUTPUT = (
    f"{OUTPUT_DIR}/radar_comparison.png"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

fedavg = pd.read_csv(
    FEDAVG_RESULTS
)

fedprox = pd.read_csv(
    FEDPROX_RESULTS
)


fedavg["Model"] = "FedAvg DenseNet-121"

fedprox["Model"] = "FedProx DenseNet-121"


comparison = pd.concat(
    [
        fedavg,
        fedprox
    ],
    ignore_index=True
)


# reorder columns

comparison = comparison[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1-score",
        "ROC-AUC"
    ]
]


comparison.to_csv(
    CSV_OUTPUT,
    index=False
)


print("="*70)
print("MODEL COMPARISON")
print("="*70)

print(comparison)


# ============================================================
# PERFORMANCE BAR CHART
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1-score",
    "ROC-AUC"
]


x = np.arange(
    len(metrics)
)

width = 0.35


plt.figure(
    figsize=(12,6)
)


plt.bar(
    x-width/2,
    comparison.iloc[0][metrics],
    width,
    label="FedAvg"
)


plt.bar(
    x+width/2,
    comparison.iloc[1][metrics],
    width,
    label="FedProx"
)


plt.xticks(
    x,
    metrics,
    rotation=45
)


plt.ylabel(
    "Score"
)


plt.title(
    "FedAvg vs FedProx DenseNet-121 Performance Comparison"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    BAR_OUTPUT,
    dpi=300
)


plt.close()



# ============================================================
# ROC-AUC COMPARISON
# ============================================================

plt.figure(
    figsize=(6,5)
)


models = comparison["Model"]

scores = comparison["ROC-AUC"]


plt.bar(
    models,
    scores
)


plt.ylabel(
    "ROC-AUC"
)


plt.title(
    "ROC-AUC Comparison"
)


plt.xticks(
    rotation=20
)


plt.ylim(
    0,
    1
)


plt.tight_layout()


plt.savefig(
    ROC_OUTPUT,
    dpi=300
)


plt.close()



# ============================================================
# RADAR CHART
# ============================================================


labels = metrics

fedavg_values = comparison.iloc[0][metrics].values.tolist()

fedprox_values = comparison.iloc[1][metrics].values.tolist()


fedavg_values += fedavg_values[:1]

fedprox_values += fedprox_values[:1]


angles = np.linspace(
    0,
    2*np.pi,
    len(labels)+1
)


fig = plt.figure(
    figsize=(7,7)
)


ax = fig.add_subplot(
    111,
    polar=True
)


ax.plot(
    angles,
    fedavg_values,
    label="FedAvg"
)


ax.plot(
    angles,
    fedprox_values,
    label="FedProx"
)


ax.set_xticks(
    angles[:-1]
)

ax.set_xticklabels(
    labels
)


ax.set_ylim(
    0,
    1
)


ax.legend(
    loc="upper right"
)


plt.title(
    "Federated Model Metric Comparison"
)


plt.tight_layout()


plt.savefig(
    RADAR_OUTPUT,
    dpi=300
)


plt.close()



print("\nSaved:")
print(CSV_OUTPUT)
print(BAR_OUTPUT)
print(ROC_OUTPUT)
print(RADAR_OUTPUT)

print("\nComparison completed.")