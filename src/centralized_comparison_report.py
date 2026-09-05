import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = "results"

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "centralized_comparison_report.csv"
)

OUTPUT_TXT = os.path.join(
    RESULTS_DIR,
    "centralized_comparison_summary.txt"
)


# ============================================================
# CENTRALIZED RESULTS
# ============================================================

results = [

    # --------------------------------------------------------
    # ResNet-18
    # --------------------------------------------------------

    {
        "Model": "ResNet-18",
        "Threshold": 0.50,
        "Accuracy": 0.759287,
        "Precision": 0.968100,
        "Sensitivity": 0.698699,
        "Specificity": 0.933718,
        "F1-score": 0.811628,
        "ROC-AUC": 0.907260
    },

    {
        "Model": "ResNet-18",
        "Threshold": 0.10,
        "Accuracy": 0.817236,
        "Precision": 0.936269,
        "Sensitivity": 0.808809,
        "Specificity": 0.841499,
        "F1-score": 0.867884,
        "ROC-AUC": 0.907260
    },


    # --------------------------------------------------------
    # DenseNet-121
    # --------------------------------------------------------

    {
        "Model": "DenseNet-121",
        "Threshold": 0.50,
        "Accuracy": 0.846211,
        "Precision": 0.937086,
        "Sensitivity": 0.849850,
        "Specificity": 0.835735,
        "F1-score": 0.891339,
        "ROC-AUC": 0.920921
    },

    {
        "Model": "DenseNet-121",
        "Threshold": 0.07,
        "Accuracy": 0.878900,
        "Precision": 0.895833,
        "Sensitivity": 0.946947,
        "Specificity": 0.682997,
        "F1-score": 0.920681,
        "ROC-AUC": 0.920921
    },


    # --------------------------------------------------------
    # EfficientNet-B0
    # --------------------------------------------------------

    {
        "Model": "EfficientNet-B0",
        "Threshold": 0.50,
        "Accuracy": 0.720654,
        "Precision": 0.967016,
        "Sensitivity": 0.645646,
        "Specificity": 0.936599,
        "F1-score": 0.774310,
        "ROC-AUC": 0.891067
    },

    {
        "Model": "EfficientNet-B0",
        "Threshold": 0.01,
        "Accuracy": 0.811293,
        "Precision": 0.898396,
        "Sensitivity": 0.840841,
        "Specificity": 0.726225,
        "F1-score": 0.868666,
        "ROC-AUC": 0.891067
    }

]


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(results)


# ============================================================
# SAVE CSV
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PRINT REPORT
# ============================================================

print("=" * 70)
print("CENTRALIZED MODEL COMPARISON REPORT")
print("=" * 70)

print()

print(df)


# ============================================================
# RANKING BY F1
# ============================================================

ranking = (
    df[df["Threshold"] != 0.50]
    .sort_values(
        by="F1-score",
        ascending=False
    )
    .reset_index(drop=True)
)


ranking.insert(
    0,
    "Rank",
    range(1, len(ranking)+1)
)


print("\n" + "=" * 70)
print("OPTIMIZED THRESHOLD RANKING")
print("=" * 70)

print(
    ranking[
        [
            "Rank",
            "Model",
            "Threshold",
            "F1-score",
            "ROC-AUC",
            "Sensitivity",
            "Specificity"
        ]
    ]
)


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

with open(
    OUTPUT_TXT,
    "w"
) as f:

    f.write(
        "Centralized Model Comparison Summary\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

    f.write(
        df.to_string()
    )

    f.write(
        "\n\nOptimized Threshold Ranking\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

    f.write(
        ranking.to_string()
    )


print("\n" + "=" * 70)
print("REPORT GENERATED")
print("=" * 70)

print()

print(
    "CSV saved:"
)

print(
    os.path.abspath(OUTPUT_CSV)
)

print()

print(
    "Summary saved:"
)

print(
    os.path.abspath(OUTPUT_TXT)
)