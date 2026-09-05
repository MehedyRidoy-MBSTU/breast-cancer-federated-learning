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

OUTPUT_DIR = PROJECT_ROOT / "results" / "gradcam"
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

print("=" * 60)
print("Grad-CAM — ResNet-18")
print("=" * 60)
print(f"Device: {DEVICE}")


# ============================================================
# Load model
# ============================================================

print("\nLoading model...")

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
# Load predictions
# ============================================================

df = pd.read_csv(PREDICTIONS)

print(f"\nTest prediction records: {len(df)}")

print("\nPrediction columns:")
print(df.columns.tolist())


# ============================================================
# Identify prediction columns
# ============================================================

# Expected columns from our evaluation script:
# true_label
# predicted_label
# malignant_probability
# path

required_columns = [
    "true_label",
    "predicted_label",
    "malignant_probability",
    "path",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' not found in prediction CSV."
        )


# ============================================================
# Define cases
# ============================================================

# True Negative:
# actual benign (0), predicted benign (0)

# True Positive:
# actual malignant (1), predicted malignant (1)

# False Negative:
# actual malignant (1), predicted benign (0)

# False Positive:
# actual benign (0), predicted malignant (1)

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


print("\nAvailable cases:")
print(f"True Negatives : {len(tn)}")
print(f"True Positives : {len(tp)}")
print(f"False Negatives: {len(fn)}")
print(f"False Positives: {len(fp)}")


# ============================================================
# Select representative examples
# ============================================================

selected = []

# ------------------------------------------------------------
# 1. Very confident benign correct prediction
# ------------------------------------------------------------

if len(tn) > 0:
    example = tn.sort_values(
        "malignant_probability",
        ascending=True
    ).iloc[0].copy()

    example["case"] = "TN_1"
    example["case_description"] = (
        "Benign correctly classified — high confidence"
    )

    selected.append(example)


# ------------------------------------------------------------
# 2. Another benign correct prediction
# ------------------------------------------------------------

if len(tn) > 1:
    example = tn.sort_values(
        "malignant_probability",
        ascending=False
    ).iloc[0].copy()

    example["case"] = "TN_2"
    example["case_description"] = (
        "Benign correctly classified — moderate confidence"
    )

    selected.append(example)


# ------------------------------------------------------------
# 3. Very confident malignant correct prediction
# ------------------------------------------------------------

if len(tp) > 0:
    example = tp.sort_values(
        "malignant_probability",
        ascending=False
    ).iloc[0].copy()

    example["case"] = "TP_1"
    example["case_description"] = (
        "Malignant correctly classified — high confidence"
    )

    selected.append(example)


# ------------------------------------------------------------
# 4. Another malignant correct prediction
# ------------------------------------------------------------

if len(tp) > 1:
    example = tp.sort_values(
        "malignant_probability",
        ascending=True
    ).iloc[0].copy()

    example["case"] = "TP_2"
    example["case_description"] = (
        "Malignant correctly classified — moderate confidence"
    )

    selected.append(example)


# ------------------------------------------------------------
# 5. False Negative
# ------------------------------------------------------------

if len(fn) > 0:
    example = fn.sort_values(
        "malignant_probability",
        ascending=False
    ).iloc[0].copy()

    example["case"] = "FN_1"
    example["case_description"] = (
        "Malignant incorrectly classified as benign"
    )

    selected.append(example)


# ------------------------------------------------------------
# 6. False Positive
# ------------------------------------------------------------

if len(fp) > 0:
    example = fp.sort_values(
        "malignant_probability",
        ascending=True
    ).iloc[0].copy()

    example["case"] = "FP_1"
    example["case_description"] = (
        "Benign incorrectly classified as malignant"
    )

    selected.append(example)


selected_df = pd.DataFrame(selected)


# ============================================================
# Save selected cases
# ============================================================

selected_csv = OUTPUT_DIR / "gradcam_selected_cases.csv"

selected_df.to_csv(
    selected_csv,
    index=False
)

print("\nSelected Grad-CAM cases:")
print(
    selected_df[
        [
            "case",
            "case_description",
            "true_label",
            "predicted_label",
            "malignant_probability",
        ]
    ].to_string(index=False)
)

print(f"\nSaved selection to:")
print(selected_csv)


# ============================================================
# Evaluation transform
# ============================================================

transform = get_eval_transforms()


# ============================================================
# Grad-CAM target layer
# ============================================================

# For ResNet-18, layer4[-1] is the final residual block
# before global average pooling and the fully connected layer.

target_layers = [model.layer4[-1]]


# ============================================================
# Grad-CAM generation
# ============================================================

cam = GradCAM(
    model=model,
    target_layers=target_layers
)


# ============================================================
# Process each selected case
# ============================================================

for _, row in selected_df.iterrows():

    case_name = row["case"]

    image_path = Path(row["path"])

    true_label = int(row["true_label"])
    predicted_label = int(row["predicted_label"])
    malignant_probability = float(
        row["malignant_probability"]
    )

    print("\n" + "-" * 60)
    print(f"Processing {case_name}")
    print(f"Image: {image_path.name}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_label}")
    print(f"Malignant probability: {malignant_probability:.4f}")

    # --------------------------------------------------------
    # Load original image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    # Display image at model input resolution
    image_resized = image.resize((224, 224))

    rgb_image = np.asarray(
        image_resized
    ).astype(np.float32) / 255.0

    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    # Explain the class predicted by the model.
    targets = [
        ClassifierOutputTarget(predicted_label)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    true_class_name = (
        "Benign" if true_label == 0
        else "Malignant"
    )

    predicted_class_name = (
        "Benign" if predicted_label == 0
        else "Malignant"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Original
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Grad-CAM
    axes[1].imshow(
        grayscale_cam,
        cmap="jet"
    )
    axes[1].set_title(
        "Grad-CAM Heatmap"
    )
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(visualization)
    axes[2].set_title(
        "Grad-CAM Overlay"
    )
    axes[2].axis("off")

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        f"{case_name} | "
        f"True: {true_class_name} | "
        f"Predicted: {predicted_class_name} | "
        f"P(Malignant): {malignant_probability:.3f}",
        fontsize=12
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR /
        f"{case_name}_gradcam.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")


# ============================================================
# Create summary figure
# ============================================================

print("\nCreating summary figure...")

fig, axes = plt.subplots(
    len(selected_df),
    3,
    figsize=(15, 5 * len(selected_df))
)

if len(selected_df) == 1:
    axes = np.expand_dims(axes, axis=0)


for row_idx, (_, row) in enumerate(selected_df.iterrows()):

    case_name = row["case"]

    image_path = Path(row["path"])

    true_label = int(row["true_label"])
    predicted_label = int(row["predicted_label"])

    image = Image.open(
        image_path
    ).convert("RGB")

    image_resized = image.resize(
        (224, 224)
    )

    rgb_image = (
        np.asarray(image_resized)
        .astype(np.float32) / 255.0
    )

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    targets = [
        ClassifierOutputTarget(predicted_label)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    true_name = (
        "Benign" if true_label == 0
        else "Malignant"
    )

    pred_name = (
        "Benign" if predicted_label == 0
        else "Malignant"
    )

    axes[row_idx, 0].imshow(rgb_image)
    axes[row_idx, 0].set_title(
        f"{case_name} — Original"
    )
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(
        grayscale_cam,
        cmap="jet"
    )
    axes[row_idx, 1].set_title(
        f"{case_name} — Grad-CAM"
    )
    axes[row_idx, 1].axis("off")

    axes[row_idx, 2].imshow(
        visualization
    )
    axes[row_idx, 2].set_title(
        f"True: {true_name} | "
        f"Pred: {pred_name}"
    )
    axes[row_idx, 2].axis("off")


fig.suptitle(
    "Grad-CAM Explainability Analysis — ResNet-18",
    fontsize=16
)

plt.tight_layout()

summary_file = (
    OUTPUT_DIR /
    "gradcam_summary.png"
)

plt.savefig(
    summary_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nSummary saved to:")
print(summary_file)

print("\n" + "=" * 60)
print("Grad-CAM analysis completed successfully.")
print("=" * 60)