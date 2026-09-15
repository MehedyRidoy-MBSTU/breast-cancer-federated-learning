import os
import numpy as np
import pandas as pd
import torch

from PIL import Image

from torchvision.transforms.functional import to_pil_image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


from src.models.densenet121 import create_densenet121
from src.data.breakhis_dataset import BreaKHisDataset
from src.data.transforms import get_eval_transforms



# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224


TEST_CSV = (
    "processed/splits/test.csv"
)


CHECKPOINT = (
    "checkpoints/"
    "scaffold_densenet121_final.pth"
)


OUTPUT_DIR = (
    "results/gradcam_scaffold_densenet121"
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
print("GRAD-CAM SCAFFOLD DENSENET-121")
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
    "SCAFFOLD DenseNet-121 loaded"
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
# GRAD CAM
# ============================================================


target_layers = [
    model.features[-1]
]


cam = GradCAM(
    model=model,
    target_layers=target_layers
)



# ============================================================
# FIND TP TN FP FN
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

    for idx in range(len(dataset)):

        image, label = dataset[idx]


        input_tensor = (
            image
            .unsqueeze(0)
            .to(DEVICE)
        )


        output = model(
            input_tensor
        )


        prediction = torch.argmax(
            output,
            dim=1
        ).item()



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



print(
    cases
)



# ============================================================
# GENERATE CAM
# ============================================================


summary = []



for case_name, idx in cases.items():

    if idx is None:

        continue


    image, label = dataset[idx]


    input_tensor = (
        image
        .unsqueeze(0)
        .to(DEVICE)
    )


    output = model(
        input_tensor
    )


    prediction = torch.argmax(
        output,
        dim=1
    ).item()



    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[
            ClassifierOutputTarget(
                prediction
            )
        ]
    )[0]



    rgb_img = (
        image
        .permute(1,2,0)
        .cpu()
        .numpy()
    )


    rgb_img = np.clip(
        rgb_img,
        0,
        1
    )


    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )



    case_dir = os.path.join(
        OUTPUT_DIR,
        case_name
    )


    os.makedirs(
        case_dir,
        exist_ok=True
    )



    original = to_pil_image(
        image
    )


    original.save(
        os.path.join(
            case_dir,
            f"{case_name}_original.png"
        )
    )


    Image.fromarray(
        visualization
    ).save(
        os.path.join(
            case_dir,
            f"{case_name}_heatmap.png"
        )
    )



    summary.append(

        {

        "case": case_name,

        "index": idx,

        "true_label": int(label),

        "prediction": int(prediction)

        }

    )



pd.DataFrame(summary).to_csv(

    os.path.join(
        OUTPUT_DIR,
        "scaffold_densenet121_gradcam_cases.csv"
    ),

    index=False

)



print("\n")
print("="*70)
print("SCAFFOLD GRAD-CAM COMPLETED")
print("="*70)