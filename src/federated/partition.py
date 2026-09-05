import os
import random
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

INPUT_CSV = "processed/splits/train.csv"

OUTPUT_DIR = "processed/federated"

NUM_CLIENTS = 5


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FEDERATED DATASET PARTITIONING")
print("=" * 70)

print(f"Input CSV: {INPUT_CSV}")
print(f"Number of clients: {NUM_CLIENTS}")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)


print("\nDataset loaded")
print(f"Total training images: {len(df)}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "specimen_id",
    "label"
]


missing = [
    col for col in required_columns
    if col not in df.columns
]


if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# SPECIMEN GROUPING
# ============================================================

specimens = (
    df["specimen_id"]
    .unique()
    .tolist()
)


print(
    f"Unique specimens: {len(specimens)}"
)


# Shuffle specimens

random.shuffle(specimens)


# ============================================================
# CREATE CLIENT SPLITS
# ============================================================

client_specimens = {
    f"client_{i+1}": []
    for i in range(NUM_CLIENTS)
}


for index, specimen in enumerate(specimens):

    client_name = (
        f"client_{(index % NUM_CLIENTS)+1}"
    )

    client_specimens[client_name].append(
        specimen
    )


# ============================================================
# SAVE CLIENT CSV FILES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


summary = []


for client_name, specimen_list in client_specimens.items():

    client_df = df[
        df["specimen_id"]
        .isin(specimen_list)
    ].copy()


    output_file = os.path.join(
        OUTPUT_DIR,
        f"{client_name}.csv"
    )


    client_df.to_csv(
        output_file,
        index=False
    )


    benign_count = (
        client_df["label"] == 0
    ).sum()


    malignant_count = (
        client_df["label"] == 1
    ).sum()


    summary.append(
        {
            "client": client_name,
            "images": len(client_df),
            "specimens": len(specimen_list),
            "benign": benign_count,
            "malignant": malignant_count,
            "malignant_ratio":
                malignant_count / len(client_df)
        }
    )


    print("\n" + "-" * 50)

    print(client_name)

    print(
        f"Images: {len(client_df)}"
    )

    print(
        f"Specimens: {len(specimen_list)}"
    )

    print(
        f"Benign: {benign_count}"
    )

    print(
        f"Malignant: {malignant_count}"
    )



# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)


summary_file = os.path.join(
    OUTPUT_DIR,
    "partition_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FEDERATED PARTITIONING COMPLETED")
print("=" * 70)


print("\nSaved files:")

for file in sorted(
    os.listdir(OUTPUT_DIR)
):
    print(
        f"  {file}"
    )


print("\nPartition Summary:")
print(summary_df)