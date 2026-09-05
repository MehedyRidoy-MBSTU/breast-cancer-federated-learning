import os
import numpy as np
import pandas as pd
import torch

import cv2

from PIL import Image

from torchvision.transforms.functional import to_pil_image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget
)


from src.models.densenet121 import create_densenet121

from src.data.breakhis_dataset import BreaKHisDataset

from src.data.transforms import (
    get_eval_transforms
)


# ============================================================
# CONFIGURATION
# ============================================================


IMAGE_SIZE = 224


TEST_CSV = (
    "processed/splits/test.csv"
)


CHECKPOINT = (
    "checkpoints/"
    "fedavg_densenet121_final.pth"
)


OUTPUT_DIR = (
    "results/gradcam"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================


print("="*70)
print("GRAD-CAM FEDAVG DENSENET-121")
print("="*70)


print(
    f"Device: {DEVICE}"
)


model = create_densenet121(
    num_classes=2
)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint
)


model.to(
    DEVICE
)

model.eval()


print(
    "FedAvg DenseNet-121 loaded"
)


# ============================================================
# DATASET
# ============================================================


dataset = BreaKHisDataset(
    TEST_CSV,
    transform=get_eval_transforms(
        IMAGE_SIZE
    )
)


print(
    f"Test images: {len(dataset)}"
)



# ============================================================
# GRAD CAM TARGET LAYER
# ============================================================


target_layers = [
    model.features[-1]
]


cam = GradCAM(
    model=model,
    target_layers=target_layers
)



# ============================================================
# FIND CASES
# ============================================================


cases = {

    "TP": None,
    "TN": None,
    "FP": None,
    "FN": None

}



print(
    "Searching prediction cases..."
)



with torch.no_grad():

    for idx in range(
        len(dataset)
    ):

        image, label = dataset[idx]


        input_tensor = (
            image
            .unsqueeze(0)
            .to(DEVICE)
        )


        output = model(
            input_tensor
        )


        prediction = (
            torch.argmax(
                output,
                dim=1
            )
            .item()
        )


        label = int(label)


        if label == 1 and prediction == 1:
            key = "TP"


        elif label == 0 and prediction == 0:
            key = "TN"


        elif label == 0 and prediction == 1:
            key = "FP"


        else:
            key = "FN"


        if cases[key] is None:

            cases[key] = idx



        if all(
            v is not None
            for v in cases.values()
        ):
            break



print(
    cases
)



# ============================================================
# GENERATE CAM IMAGES
# ============================================================


for case, idx in cases.items():

    print(
        f"Generating {case}"
    )


    image, label = dataset[idx]


    input_tensor = (
        image
        .unsqueeze(0)
        .to(DEVICE)
    )


    output = model(
        input_tensor
    )


    prediction = (
        torch.argmax(
            output,
            dim=1
        )
        .item()
    )


    targets = [
        ClassifierOutputTarget(
            prediction
        )
    ]


    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )


    grayscale_cam = (
        grayscale_cam[0]
    )


    rgb_img = (
        image
        .permute(
            1,
            2,
            0
        )
        .cpu()
        .numpy()
    )


    rgb_img = (
        rgb_img -
        rgb_img.min()
    ) / (
        rgb_img.max()
        -
        rgb_img.min()
    )


    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )


    save_dir = os.path.join(
        OUTPUT_DIR,
        case
    )


    os.makedirs(
        save_dir,
        exist_ok=True
    )


    cv2.imwrite(
        os.path.join(
            save_dir,
            f"{case}_heatmap.png"
        ),
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )


    original = (
        rgb_img * 255
    ).astype(
        np.uint8
    )


    cv2.imwrite(
        os.path.join(
            save_dir,
            f"{case}_original.png"
        ),
        cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR
        )
    )


print("="*70)

print(
    "GRAD-CAM COMPLETED"
)

print("="*70)