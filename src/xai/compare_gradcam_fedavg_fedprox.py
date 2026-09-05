import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

FEDAVG_DIR = Path("results/gradcam_fedavg_densenet121")
FEDPROX_DIR = Path("results/gradcam_fedprox_densenet121")

OUTPUT_DIR = Path("results/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "gradcam_fedavg_vs_fedprox.png"

CASES = ["TP", "TN", "FP", "FN"]


# ============================================================
# IMAGE PATH BUILDER
# ============================================================

def get_image_paths(base_dir, case_name):
    original_path = base_dir / case_name / f"{case_name}_original.png"
    heatmap_path = base_dir / case_name / f"{case_name}_heatmap.png"
    return original_path, heatmap_path


# ============================================================
# VALIDATION
# ============================================================

print("=" * 70)
print("GRAD-CAM COMPARISON: FEDAVG VS FEDPROX")
print("=" * 70)

for case_name in CASES:
    fedavg_original, fedavg_heatmap = get_image_paths(FEDAVG_DIR, case_name)
    fedprox_original, fedprox_heatmap = get_image_paths(FEDPROX_DIR, case_name)

    for path in [
        fedavg_original,
        fedavg_heatmap,
        fedprox_original,
        fedprox_heatmap,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

print("All required Grad-CAM files found.")


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    nrows=4,
    ncols=4,
    figsize=(16, 16)
)

column_titles = [
    "FedAvg Original",
    "FedAvg Heatmap",
    "FedProx Original",
    "FedProx Heatmap",
]

for col_idx, title in enumerate(column_titles):
    axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold")

for row_idx, case_name in enumerate(CASES):
    fedavg_original, fedavg_heatmap = get_image_paths(FEDAVG_DIR, case_name)
    fedprox_original, fedprox_heatmap = get_image_paths(FEDPROX_DIR, case_name)

    images = [
        Image.open(fedavg_original).convert("RGB"),
        Image.open(fedavg_heatmap).convert("RGB"),
        Image.open(fedprox_original).convert("RGB"),
        Image.open(fedprox_heatmap).convert("RGB"),
    ]

    for col_idx, img in enumerate(images):
        axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].axis("off")

    axes[row_idx, 0].set_ylabel(
        case_name,
        fontsize=12,
        fontweight="bold",
        rotation=90,
        labelpad=20
    )

plt.suptitle(
    "Grad-CAM Comparison: FedAvg vs FedProx (DenseNet-121)",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("GRAD-CAM COMPARISON FIGURE GENERATED")
print("=" * 70)
print(f"Saved: {OUTPUT_FILE}")