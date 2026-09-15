import os
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shap

from src.models.densenet121 import create_densenet121
from src.data.breakhis_dataset import BreaKHisDataset
from src.data.transforms import get_eval_transforms


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224

BACKGROUND_PER_CLASS = 5

SHAP_SAMPLES = 50

TEST_CSV = "processed/splits/test.csv"


CHECKPOINTS = {

    "fedavg":
        "checkpoints/fedavg_densenet121_final.pth",

    "fedprox":
        "checkpoints/fedprox_densenet121_final.pth",

    "scaffold":
        "checkpoints/scaffold_densenet121_final.pth",

}


OUTPUT_BASE = "results"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


# ============================================================
# COMMAND LINE ARGUMENT
# ============================================================

if len(sys.argv) != 2:

    print(
        "Usage:\n"
        "python -m src.xai.shap_densenet121 "
        "fedavg|fedprox|scaffold"
    )

    sys.exit(1)


MODEL_NAME = sys.argv[1].lower()


if MODEL_NAME not in CHECKPOINTS:

    raise ValueError(
        "Model must be one of: "
        "fedavg, fedprox, scaffold"
    )


CHECKPOINT = CHECKPOINTS[MODEL_NAME]


OUTPUT_DIR = os.path.join(

    OUTPUT_BASE,

    f"shap_{MODEL_NAME}_densenet121"

)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


SUMMARY_FILE = os.path.join(

    OUTPUT_DIR,

    f"{MODEL_NAME}_densenet121_shap_cases.csv"

)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    f"SHAP DENSENET-121 : {MODEL_NAME.upper()}"
)

print("=" * 70)

print(
    f"Device: {DEVICE}"
)

print(
    f"Checkpoint: {CHECKPOINT}"
)


# ============================================================
# SAFE DENSENET WRAPPER
# ============================================================

class SafeDenseNet121(nn.Module):

    """
    Wrapper around torchvision DenseNet.

    Important:
    torchvision DenseNet forward() contains:

        F.relu(features, inplace=True)

    SHAP gradient methods can fail with that operation.

    This wrapper reproduces DenseNet forward propagation
    but uses inplace=False.

    The trained parameters are NOT modified.
    """

    def __init__(self, base_model):

        super().__init__()

        self.features = base_model.features

        self.classifier = base_model.classifier


    def forward(self, x):

        features = self.features(x)

        out = F.relu(
            features,
            inplace=False
        )

        out = F.adaptive_avg_pool2d(

            out,

            output_size=(1, 1)

        )

        out = torch.flatten(
            out,
            1
        )

        out = self.classifier(
            out
        )

        return out


# ============================================================
# DISABLE MODULE-LEVEL INPLACE RELU
# ============================================================

def disable_inplace_relu(model):

    count = 0

    for module in model.modules():

        if isinstance(
            module,
            nn.ReLU
        ):

            if module.inplace:

                module.inplace = False

                count += 1

    return count


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    base_model = create_densenet121(
        num_classes=2
    )


    checkpoint = torch.load(

        CHECKPOINT,

        map_location="cpu"

    )


    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint


    base_model.load_state_dict(
        state_dict
    )


    relu_count = disable_inplace_relu(
        base_model
    )


    safe_model = SafeDenseNet121(
        base_model
    )


    safe_model = safe_model.to(
        DEVICE
    )


    safe_model.eval()


    return safe_model, relu_count


model, relu_count = load_model()


print(
    "Model loaded successfully"
)

print(
    f"Module-level in-place ReLU disabled: {relu_count}"
)

print(
    "DenseNet final functional ReLU: inplace=False"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

dataset = BreaKHisDataset(

    TEST_CSV,

    transform=get_eval_transforms(
        IMAGE_SIZE
    )

)


test_df = pd.read_csv(
    TEST_CSV
)


print(
    f"Test images: {len(dataset)}"
)


# ============================================================
# FIND TP / TN / FP / FN
# ============================================================

cases = {

    "TP": None,

    "TN": None,

    "FP": None,

    "FN": None

}


print(
    "\nSearching TP/TN/FP/FN cases..."
)


with torch.no_grad():

    for idx in range(
        len(dataset)
    ):

        image, label = dataset[idx]


        label = int(
            label
        )


        input_tensor = (

            image
            .unsqueeze(0)
            .to(DEVICE)

        )


        logits = model(
            input_tensor
        )


        prediction = int(

            torch.argmax(

                logits,

                dim=1

            ).item()

        )


        if (
            label == 1
            and
            prediction == 1
        ):

            case_name = "TP"


        elif (
            label == 0
            and
            prediction == 0
        ):

            case_name = "TN"


        elif (
            label == 0
            and
            prediction == 1
        ):

            case_name = "FP"


        else:

            case_name = "FN"


        if cases[case_name] is None:

            cases[case_name] = idx


        if all(

            value is not None

            for value in cases.values()

        ):

            break


print(
    "Selected cases:"
)

print(
    cases
)


# ============================================================
# BUILD BALANCED SHAP BACKGROUND
# ============================================================

print(
    "\nPreparing balanced SHAP background..."
)


background_images = []


background_counts = {

    0: 0,

    1: 0

}


for idx in range(
    len(dataset)
):

    image, label = dataset[idx]


    label = int(
        label
    )


    if (

        background_counts[label]

        <

        BACKGROUND_PER_CLASS

    ):

        background_images.append(
            image
        )


        background_counts[label] += 1


    if (

        background_counts[0]
        >=
        BACKGROUND_PER_CLASS

        and

        background_counts[1]
        >=
        BACKGROUND_PER_CLASS

    ):

        break


background = torch.stack(
    background_images
)


background = background.to(
    DEVICE
)


print(
    f"Background samples: {len(background)}"
)

print(
    f"Benign background: {background_counts[0]}"
)

print(
    f"Malignant background: {background_counts[1]}"
)


# ============================================================
# CREATE GRADIENT EXPLAINER
# ============================================================

print(
    "\nCreating SHAP GradientExplainer..."
)


explainer = shap.GradientExplainer(

    model,

    background

)


print(
    "GradientExplainer initialized successfully"
)


# ============================================================
# CONVERT MODEL INPUT TO DISPLAYABLE RGB
# ============================================================

def tensor_to_display_image(
    tensor
):

    image = (

        tensor
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()

    )


    display_image = np.zeros_like(
        image,
        dtype=np.float32
    )


    for channel in range(
        image.shape[2]
    ):

        channel_data = image[
            :,
            :,
            channel
        ]


        minimum = np.min(
            channel_data
        )


        maximum = np.max(
            channel_data
        )


        if maximum > minimum:

            display_image[
                :,
                :,
                channel
            ] = (

                channel_data
                -
                minimum

            ) / (

                maximum
                -
                minimum

            )


        else:

            display_image[
                :,
                :,
                channel
            ] = 0.0


    return np.clip(

        display_image,

        0.0,

        1.0

    )


# ============================================================
# EXTRACT CLASS-SPECIFIC SHAP VALUES
# ============================================================

def extract_class_shap(

    shap_values,

    predicted_class

):

    if isinstance(
        shap_values,
        list
    ):

        values = np.asarray(

            shap_values[
                predicted_class
            ]

        )


        if values.ndim == 4:

            return values[0]


    values = np.asarray(
        shap_values
    )


    # SHAP 0.51 commonly returns:
    #
    # batch x channels x height x width x outputs

    if values.ndim == 5:

        if values.shape[-1] == 2:

            return values[

                0,

                :,

                :,

                :,

                predicted_class

            ]


        if values.shape[1] == 2:

            return values[

                0,

                predicted_class,

                :,

                :,

                :

            ]


    # Older / single-output behavior:
    #
    # batch x channels x height x width

    if values.ndim == 4:

        return values[0]


    raise RuntimeError(

        "Unexpected SHAP output shape: "
        f"{values.shape}"

    )


# ============================================================
# CREATE 2D SHAP HEATMAP
# ============================================================

def create_shap_map(
    class_values
):

    values = np.asarray(
        class_values
    )


    if values.ndim == 3:

        # Channels first:
        # C x H x W

        if values.shape[0] in (
            1,
            3
        ):

            shap_map = np.sum(

                values,

                axis=0

            )


        # Channels last:
        # H x W x C

        elif values.shape[-1] in (
            1,
            3
        ):

            shap_map = np.sum(

                values,

                axis=-1

            )


        else:

            raise RuntimeError(

                "Unexpected 3D SHAP shape: "
                f"{values.shape}"

            )


    elif values.ndim == 2:

        shap_map = values


    else:

        raise RuntimeError(

            "Cannot create SHAP map from shape: "
            f"{values.shape}"

        )


    return shap_map


# ============================================================
# GENERATE SHAP EXPLANATIONS
# ============================================================

summary_rows = []


print(
    "\nGenerating SHAP explanations..."
)


for case_name, idx in cases.items():

    if idx is None:

        print(
            f"\nSkipping {case_name}: "
            "no case found."
        )

        continue


    print(
        f"\nProcessing {case_name} "
        f"(dataset index {idx})..."
    )


    image, label = dataset[
        idx
    ]


    label = int(
        label
    )


    input_tensor = (

        image
        .unsqueeze(0)
        .to(DEVICE)

    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            input_tensor
        )


        probabilities = torch.softmax(

            logits,

            dim=1

        )


        prediction = int(

            torch.argmax(

                probabilities,

                dim=1

            ).item()

        )


        benign_probability = float(

            probabilities[
                0,
                0
            ]
            .detach()
            .cpu()
            .item()

        )


        malignant_probability = float(

            probabilities[
                0,
                1
            ]
            .detach()
            .cpu()
            .item()

        )


    # --------------------------------------------------------
    # COMPUTE SHAP
    # --------------------------------------------------------

    shap_values = explainer.shap_values(

        input_tensor,

        nsamples=SHAP_SAMPLES,

        rseed=SEED

    )


    class_values = extract_class_shap(

        shap_values,

        prediction

    )


    shap_map = create_shap_map(

        class_values

    )


    print(
        f"SHAP map shape: {shap_map.shape}"
    )


    # --------------------------------------------------------
    # HEATMAP SCALE
    # --------------------------------------------------------

    max_abs = float(

        np.max(

            np.abs(
                shap_map
            )

        )

    )


    if (
        not np.isfinite(max_abs)
        or
        max_abs == 0
    ):

        max_abs = 1.0


    # --------------------------------------------------------
    # CASE OUTPUT DIRECTORY
    # --------------------------------------------------------

    case_dir = os.path.join(

        OUTPUT_DIR,

        case_name

    )


    os.makedirs(

        case_dir,

        exist_ok=True

    )


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    display_image = tensor_to_display_image(
        image
    )


    # --------------------------------------------------------
    # SAVE ORIGINAL
    # --------------------------------------------------------

    original_path = os.path.join(

        case_dir,

        f"{case_name}_original.png"

    )


    plt.figure(
        figsize=(6, 6)
    )


    plt.imshow(
        display_image
    )


    plt.axis(
        "off"
    )


    plt.tight_layout()


    plt.savefig(

        original_path,

        dpi=300,

        bbox_inches="tight",

        pad_inches=0

    )


    plt.close()


    # --------------------------------------------------------
    # SAVE SHAP OVERLAY
    # --------------------------------------------------------

    shap_path = os.path.join(

        case_dir,

        f"{case_name}_shap.png"

    )


    plt.figure(
        figsize=(6, 6)
    )


    plt.imshow(
        display_image
    )


    heatmap = plt.imshow(

        shap_map,

        cmap="seismic",

        alpha=0.50,

        vmin=-max_abs,

        vmax=max_abs

    )


    plt.colorbar(

        heatmap,

        fraction=0.046,

        pad=0.04,

        label="SHAP value"

    )


    plt.title(

        (
            f"{MODEL_NAME.upper()} "
            f"DenseNet-121 SHAP - {case_name}\n"
            f"True={label} | "
            f"Predicted={prediction} | "
            f"P(Malignant)="
            f"{malignant_probability:.4f}"
        ),

        fontsize=10

    )


    plt.axis(
        "off"
    )


    plt.tight_layout()


    plt.savefig(

        shap_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    # --------------------------------------------------------
    # SAVE SUMMARY ROW
    # --------------------------------------------------------

    summary_rows.append(

        {

            "case":
                case_name,

            "dataset_index":
                idx,

            "true_label":
                label,

            "prediction":
                prediction,

            "benign_probability":
                benign_probability,

            "malignant_probability":
                malignant_probability,

            "shap_explainer":
                "GradientExplainer",

            "background_size":
                len(background_images),

            "shap_samples":
                SHAP_SAMPLES

        }

    )


    print(
        f"{case_name} completed"
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(

    SUMMARY_FILE,

    index=False

)


# ============================================================
# FINISH
# ============================================================

print("\n")

print("=" * 70)

print(
    "SHAP COMPLETED"
)

print("=" * 70)

print(
    f"Model: {MODEL_NAME.upper()}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print(
    f"Summary CSV: {SUMMARY_FILE}"
)