from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data.transforms import get_eval_transforms
from src.models.resnet18 import create_resnet18


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "resnet18_best.pth"
PREDICTIONS = PROJECT_ROOT / "results" / "resnet18_test_predictions.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "gradcam_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("Grad-CAM v2 — ResNet-18")
print("=" * 70)
print(f"Device: {DEVICE}")


# ============================================================
# Load model
# ============================================================

print("\nLoading ResNet-18...")

model = create_resnet18(num_classes=2)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# Load test predictions
# ============================================================

df = pd.read_csv(PREDICTIONS)

print(f"Test images: {len(df)}")


required_columns = [
    "path",
    "filename",
    "true_label",
    "predicted_label",
    "malignant_probability",
    "magnification",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing."
        )


# ============================================================
# Create confusion-matrix groups
# ============================================================

tn = df[
    (df["true_label"] == 0) &
    (df["predicted_label"] == 0)
].copy()

tp = df[
    (df["true_label"] == 1) &
    (df["predicted_label"] == 1)
].copy()

fn = df[
    (df["true_label"] == 1) &
    (df["predicted_label"] == 0)
].copy()

fp = df[
    (df["true_label"] == 0) &
    (df["predicted_label"] == 1)
].copy()


print("\nConfusion groups:")
print(f"TN: {len(tn)}")
print(f"TP: {len(tp)}")
print(f"FN: {len(fn)}")
print(f"FP: {len(fp)}")


# ============================================================
# Select representative examples
# ============================================================

selected = []


def add_case(dataframe, case_name, description, sort_column,
             ascending, number=1):

    if len(dataframe) == 0:
        return

    sorted_df = dataframe.sort_values(
        sort_column,
        ascending=ascending
    )

    existing_count = sum(
        1 for item in selected
        if str(item["case"]).startswith(case_name + "_")
    )

    for i in range(min(number, len(sorted_df))):

        row = sorted_df.iloc[i].copy()

        case_number = existing_count + i + 1

        row["case"] = f"{case_name}_{case_number}"
        row["case_description"] = description

        selected.append(row)
# ------------------------------------------------------------
# True Negatives
# ------------------------------------------------------------

# Very confident benign
add_case(
    tn,
    "TN",
    "Benign correctly classified — high benign confidence",
    "malignant_probability",
    True,
    1
)

# Borderline benign
add_case(
    tn,
    "TN",
    "Benign correctly classified — near decision boundary",
    "malignant_probability",
    False,
    1
)


# ------------------------------------------------------------
# True Positives
# ------------------------------------------------------------

# Very confident malignant
add_case(
    tp,
    "TP",
    "Malignant correctly classified — high malignant confidence",
    "malignant_probability",
    False,
    1
)

# Borderline malignant
add_case(
    tp,
    "TP",
    "Malignant correctly classified — near decision boundary",
    "malignant_probability",
    True,
    1
)


# ------------------------------------------------------------
# False Negatives
# ------------------------------------------------------------

# FN closest to decision boundary
add_case(
    fn,
    "FN",
    "Malignant incorrectly classified as benign",
    "malignant_probability",
    False,
    1
)

# FN with lowest malignant probability
add_case(
    fn,
    "FN",
    "Malignant strongly misclassified as benign",
    "malignant_probability",
    True,
    1
)


# ------------------------------------------------------------
# False Positives
# ------------------------------------------------------------

# FP closest to decision boundary
add_case(
    fp,
    "FP",
    "Benign incorrectly classified as malignant",
    "malignant_probability",
    True,
    1
)

# FP with highest malignant probability
add_case(
    fp,
    "FP",
    "Benign strongly misclassified as malignant",
    "malignant_probability",
    False,
    1
)


selected_df = pd.DataFrame(selected)


# ============================================================
# Save selected cases
# ============================================================

selection_file = OUTPUT_DIR / "selected_cases.csv"

selected_df.to_csv(
    selection_file,
    index=False
)

print("\nSelected cases:")
print(
    selected_df[
        [
            "case",
            "case_description",
            "true_label",
            "predicted_label",
            "malignant_probability",
            "magnification",
            "filename",
        ]
    ].to_string(index=False)
)

print(f"\nSaved: {selection_file}")


# ============================================================
# Transform
# ============================================================

transform = get_eval_transforms()


# ============================================================
# Target layer
# ============================================================

target_layers = [
    model.layer4[-1]
]


# ============================================================
# Grad-CAM object
# ============================================================

cam = GradCAM(
    model=model,
    target_layers=target_layers
)


# ============================================================
# Helper function
# ============================================================

def generate_gradcam(image, target_class):

    # Original image for visualization
    resized = image.resize((224, 224))

    rgb_image = (
        np.asarray(resized)
        .astype(np.float32)
        / 255.0
    )

    # Model input
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    # Target class
    targets = [
        ClassifierOutputTarget(target_class)
    ]

    # Generate CAM
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # Overlay
    overlay = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return rgb_image, grayscale_cam, overlay


# ============================================================
# Process each case
# ============================================================

for _, row in selected_df.iterrows():

    case = row["case"]

    image_path = Path(row["path"])

    true_label = int(row["true_label"])
    predicted_label = int(row["predicted_label"])

    probability = float(
        row["malignant_probability"]
    )

    print("\n" + "-" * 70)
    print(f"Processing: {case}")
    print(f"File: {image_path.name}")
    print(f"True class: {true_label}")
    print(f"Predicted class: {predicted_label}")
    print(f"P(Malignant): {probability:.6f}")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Predicted-class Grad-CAM
    # --------------------------------------------------------

    rgb_image, predicted_cam, predicted_overlay = (
        generate_gradcam(
            image,
            predicted_label
        )
    )


    # --------------------------------------------------------
    # True-class Grad-CAM
    # --------------------------------------------------------

    rgb_image, true_cam, true_overlay = (
        generate_gradcam(
            image,
            true_label
        )
    )


    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    true_name = (
        "Benign"
        if true_label == 0
        else "Malignant"
    )

    predicted_name = (
        "Benign"
        if predicted_label == 0
        else "Malignant"
    )


    # ========================================================
    # Create figure
    # ========================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10)
    )


    # --------------------------------------------------------
    # Row 1 — Predicted class
    # --------------------------------------------------------

    axes[0, 0].imshow(rgb_image)

    axes[0, 0].set_title(
        "Original Image"
    )

    axes[0, 0].axis("off")


    axes[0, 1].imshow(
        predicted_cam,
        cmap="jet"
    )

    axes[0, 1].set_title(
        f"Grad-CAM: Predicted Class\n"
        f"{predicted_name}"
    )

    axes[0, 1].axis("off")


    axes[0, 2].imshow(
        predicted_overlay
    )

    axes[0, 2].set_title(
        "Predicted-Class Overlay"
    )

    axes[0, 2].axis("off")


    # --------------------------------------------------------
    # Row 2 — True class
    # --------------------------------------------------------

    axes[1, 0].imshow(rgb_image)

    axes[1, 0].set_title(
        f"True Class: {true_name}"
    )

    axes[1, 0].axis("off")


    axes[1, 1].imshow(
        true_cam,
        cmap="jet"
    )

    axes[1, 1].set_title(
        f"Grad-CAM: True Class\n"
        f"{true_name}"
    )

    axes[1, 1].axis("off")


    axes[1, 2].imshow(
        true_overlay
    )

    axes[1, 2].set_title(
        "True-Class Overlay"
    )

    axes[1, 2].axis("off")


    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        f"{case} | "
        f"True: {true_name} | "
        f"Predicted: {predicted_name} | "
        f"P(Malignant): {probability:.4f}",
        fontsize=14
    )

    plt.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        f"{case}_gradcam_v2.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")


# ============================================================
# Create case summary
# ============================================================

summary_rows = []

for _, row in selected_df.iterrows():

    summary_rows.append({
        "case": row["case"],
        "description": row["case_description"],
        "true_class": (
            "Benign"
            if int(row["true_label"]) == 0
            else "Malignant"
        ),
        "predicted_class": (
            "Benign"
            if int(row["predicted_label"]) == 0
            else "Malignant"
        ),
        "malignant_probability": row[
            "malignant_probability"
        ],
        "magnification": row[
            "magnification"
        ],
        "filename": row["filename"],
        "path": row["path"],
    })


summary_df = pd.DataFrame(summary_rows)

summary_file = (
    OUTPUT_DIR /
    "gradcam_v2_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 70)
print("Grad-CAM v2 completed successfully.")
print("=" * 70)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {file.name}")