import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

FEDAVG_DIR = Path(
    "results/shap_fedavg_densenet121"
)

FEDPROX_DIR = Path(
    "results/shap_fedprox_densenet121"
)

SCAFFOLD_DIR = Path(
    "results/shap_scaffold_densenet121"
)


OUTPUT_DIR = Path(
    "results/comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_FILE = (
    OUTPUT_DIR
    /
    "shap_fedavg_vs_fedprox_vs_scaffold.png"
)


CASES = [
    "TP",
    "TN",
    "FP",
    "FN"
]


# ============================================================
# FILE VALIDATION
# ============================================================

print("=" * 70)

print(
    "SHAP COMPARISON: FEDAVG VS FEDPROX VS SCAFFOLD"
)

print("=" * 70)



def get_shap_path(
    base_dir,
    case
):

    return (
        base_dir
        /
        case
        /
        f"{case}_shap.png"
    )



all_dirs = [

    FEDAVG_DIR,

    FEDPROX_DIR,

    SCAFFOLD_DIR

]


for directory in all_dirs:

    if not directory.exists():

        raise FileNotFoundError(
            f"Missing directory: {directory}"
        )



for case in CASES:

    for directory in all_dirs:

        file_path = get_shap_path(
            directory,
            case
        )


        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing SHAP file: {file_path}"
            )


print(
    "All SHAP files found."
)



# ============================================================
# CREATE COMPARISON FIGURE
# ============================================================

fig, axes = plt.subplots(

    nrows=4,

    ncols=3,

    figsize=(12,16)

)



column_titles = [

    "FedAvg DenseNet-121",

    "FedProx DenseNet-121",

    "SCAFFOLD DenseNet-121"

]


for idx, title in enumerate(column_titles):

    axes[0, idx].set_title(

        title,

        fontsize=12,

        fontweight="bold"

    )



model_dirs = [

    FEDAVG_DIR,

    FEDPROX_DIR,

    SCAFFOLD_DIR

]



for row, case in enumerate(CASES):


    for col, directory in enumerate(model_dirs):


        image_path = get_shap_path(

            directory,

            case

        )


        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        axes[row, col].imshow(
            image
        )


        axes[row, col].axis(
            "off"
        )


    axes[row, 0].set_ylabel(

        case,

        fontsize=12,

        fontweight="bold",

        rotation=90,

        labelpad=20

    )



plt.suptitle(

    "SHAP Explanation Comparison: FedAvg vs FedProx vs SCAFFOLD (DenseNet-121)",

    fontsize=16,

    fontweight="bold"

)



plt.tight_layout(

    rect=[
        0,
        0,
        1,
        0.96
    ]

)



plt.savefig(

    OUTPUT_FILE,

    dpi=300,

    bbox_inches="tight"

)



plt.close()



# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print(
    "SHAP COMPARISON GENERATED"
)

print("=" * 70)

print(
    f"Saved: {OUTPUT_FILE}"
)