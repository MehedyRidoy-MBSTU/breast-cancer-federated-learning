import pandas as pd
import os


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = "results/comparison"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


OUTPUT_FILE = (
    f"{OUTPUT_DIR}/fedprox_mu_comparison.csv"
)


# ============================================================
# FEDPROX HYPERPARAMETER RESULTS
# ============================================================

data = [

    {
        "FedProx_mu": 0.01,
        "Accuracy": 0.2934621099,
        "Precision": 0.9800000000,
        "Sensitivity": 0.0490490490,
        "Specificity": 0.9971181556,
        "F1-score": 0.0934223070,
        "ROC-AUC": 0.6409204594,
    },


    {
        "FedProx_mu": 0.001,
        "Accuracy": 0.5118870728,
        "Precision": 0.7267904509,
        "Sensitivity": 0.5485485485,
        "Specificity": 0.4063400576,
        "F1-score": 0.6252139190,
        "ROC-AUC": 0.4708656784,
    },


    {
        "FedProx_mu": 0.0001,
        "Accuracy": 0.7273402675,
        "Precision": 0.7569105691,
        "Sensitivity": 0.9319319319,
        "Specificity": 0.1383285303,
        "F1-score": 0.8353521759,
        "ROC-AUC": 0.6643300361,
    }

]


df = pd.DataFrame(data)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("="*70)
print("FEDPROX MU COMPARISON TABLE")
print("="*70)

print(df)


print("\nSaved:")
print(OUTPUT_FILE)