import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

FEDAVG_DIR = Path(
    "results/gradcam_fedavg_densenet121"
)

FEDPROX_DIR = Path(
    "results/gradcam_fedprox_densenet121"
)

SCAFFOLD_DIR = Path(
    "results/gradcam_scaffold_densenet121"
)


OUTPUT_DIR = Path(
    "results/comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_FILE = (
    OUTPUT_DIR /
    "gradcam_fedavg_vs_fedprox_vs_scaffold.png"
)


CASES = [
    "TP",
    "TN",
    "FP",
    "FN"
]



# ============================================================
# IMAGE PATH
# ============================================================

def get_paths(base_dir, case):

    return (

        base_dir / case / f"{case}_original.png",

        base_dir / case / f"{case}_heatmap.png"

    )



# ============================================================
# VALIDATION
# ============================================================

print("="*70)

print(
    "GRAD-CAM COMPARISON: FEDAVG VS FEDPROX VS SCAFFOLD"
)

print("="*70)



for model_dir in [

    FEDAVG_DIR,

    FEDPROX_DIR,

    SCAFFOLD_DIR

]:

    for case in CASES:

        original, heatmap = get_paths(
            model_dir,
            case
        )

        for file in [
            original,
            heatmap
        ]:

            if not file.exists():

                raise FileNotFoundError(
                    f"Missing file: {file}"
                )



print(
    "All Grad-CAM files found."
)



# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(

    nrows=4,

    ncols=6,

    figsize=(18,14)

)



columns = [

    "FedAvg\nOriginal",

    "FedAvg\nHeatmap",

    "FedProx\nOriginal",

    "FedProx\nHeatmap",

    "SCAFFOLD\nOriginal",

    "SCAFFOLD\nHeatmap"

]



for i, title in enumerate(columns):

    axes[0,i].set_title(
        title,
        fontsize=11,
        fontweight="bold"
    )



for row, case in enumerate(CASES):


    images = []


    for model_dir in [

        FEDAVG_DIR,

        FEDPROX_DIR,

        SCAFFOLD_DIR

    ]:

        original, heatmap = get_paths(
            model_dir,
            case
        )


        images.extend(

            [

                Image.open(original)
                .convert("RGB"),

                Image.open(heatmap)
                .convert("RGB")

            ]

        )



    for col, img in enumerate(images):

        axes[row,col].imshow(
            img
        )

        axes[row,col].axis(
            "off"
        )



    axes[row,0].set_ylabel(

        case,

        fontsize=12,

        fontweight="bold"

    )



plt.suptitle(

    "Grad-CAM Comparison: FedAvg vs FedProx vs SCAFFOLD (DenseNet-121)",

    fontsize=16,

    fontweight="bold"

)



plt.tight_layout(
    rect=[0,0,1,0.96]
)



plt.savefig(

    OUTPUT_FILE,

    dpi=300,

    bbox_inches="tight"

)


plt.close()



print("\n")
print("="*70)
print("GRAD-CAM COMPARISON GENERATED")
print("="*70)

print(
    f"Saved: {OUTPUT_FILE}"
)