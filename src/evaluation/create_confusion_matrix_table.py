import pandas as pd
import os


OUTPUT_DIR = "results/comparison"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


OUTPUT_FILE = (
    f"{OUTPUT_DIR}/confusion_matrix_comparison.csv"
)


data = [

    {
        "Model": "FedAvg DenseNet-121",
        "TN": 294,
        "FP": 53,
        "FN": 297,
        "TP": 702
    },

    {
        "Model": "FedProx DenseNet-121",
        "TN": 48,
        "FP": 299,
        "FN": 68,
        "TP": 931
    }

]


df = pd.DataFrame(data)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("="*70)
print("CONFUSION MATRIX COMPARISON")
print("="*70)

print(df)


print("\nSaved:")
print(OUTPUT_FILE)