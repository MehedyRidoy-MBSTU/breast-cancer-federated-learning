import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FEDERATED_DIR = "processed/federated"

NUM_CLIENTS = 5


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FEDERATED PARTITION INTEGRITY CHECK")
print("=" * 70)


# ============================================================
# LOAD CLIENT FILES
# ============================================================

client_files = [
    f"client_{i}.csv"
    for i in range(1, NUM_CLIENTS + 1)
]


all_specimens = {}

total_images = 0


for file in client_files:

    path = os.path.join(
        FEDERATED_DIR,
        file
    )

    df = pd.read_csv(path)

    client_name = file.replace(
        ".csv",
        ""
    )

    print(
        f"\n{client_name}"
    )

    print(
        f"Images: {len(df)}"
    )

    specimens = set(
        df["specimen_id"]
    )

    print(
        f"Specimens: {len(specimens)}"
    )


    total_images += len(df)


    for specimen in specimens:

        if specimen in all_specimens:

            print(
                "\nWARNING: Specimen leakage detected!"
            )

            print(
                f"Specimen {specimen}"
            )

            print(
                "Found in:"
            )

            print(
                all_specimens[specimen]
            )

            print(
                client_name
            )

            raise ValueError(
                "Data leakage detected"
            )


        all_specimens[specimen] = client_name



# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("INTEGRITY CHECK COMPLETED")
print("=" * 70)


print(
    f"Total images across clients: {total_images}"
)

print(
    f"Unique specimens: {len(all_specimens)}"
)


print(
    "\nSUCCESS:"
)

print(
    "No specimen appears in multiple clients."
)