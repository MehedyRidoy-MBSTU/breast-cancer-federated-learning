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

SCAFFOLD_RESULTS = (
    "results/scaffold_densenet121_test_results.csv"
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


scaffold = pd.read_csv(
    SCAFFOLD_RESULTS
)



fedavg["Model"] = "FedAvg DenseNet-121"

fedprox["Model"] = "FedProx DenseNet-121"

scaffold["Model"] = "SCAFFOLD DenseNet-121"



comparison = pd.concat(
    [
        fedavg,
        fedprox,
        scaffold
    ],
    ignore_index=True
)



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


width = 0.25



plt.figure(
    figsize=(12,6)
)



positions = [

    x-width,

    x,

    x+width

]



models = [

    ("FedAvg", comparison.iloc[0]),

    ("FedProx", comparison.iloc[1]),

    ("SCAFFOLD", comparison.iloc[2])

]



for pos, (name, row) in zip(
    positions,
    models
):

    plt.bar(

        pos,

        row[metrics],

        width,

        label=name

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
    "FedAvg vs FedProx vs SCAFFOLD DenseNet-121 Performance Comparison"
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
    figsize=(7,5)
)



plt.bar(

    comparison["Model"],

    comparison["ROC-AUC"]

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


angles = np.linspace(

    0,

    2*np.pi,

    len(labels),

    endpoint=False

)


fig = plt.figure(

    figsize=(7,7)

)


ax = fig.add_subplot(

    111,

    polar=True

)



for _, row in comparison.iterrows():


    values = row[metrics].tolist()


    values += values[:1]


    radar_angles = np.append(

        angles,

        angles[0]

    )


    ax.plot(

        radar_angles,

        values,

        label=row["Model"]

    )



ax.set_xticks(

    angles

)


ax.set_xticklabels(

    labels

)


ax.set_ylim(

    0,

    1

)


ax.legend(

    loc="upper right",

    bbox_to_anchor=(1.3,1.1)

)



plt.title(

    "Model Performance Radar Comparison"

)



plt.tight_layout()



plt.savefig(

    RADAR_OUTPUT,

    dpi=300

)



plt.close()



print("\nComparison files saved:")
print(CSV_OUTPUT)
print(BAR_OUTPUT)
print(ROC_OUTPUT)
print(RADAR_OUTPUT)