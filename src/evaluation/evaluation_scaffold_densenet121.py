import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


from src.data.breakhis_dataset import BreaKHisDataset

from src.data.transforms import (
    get_eval_transforms
)

from src.models.densenet121 import create_densenet121



# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0


TEST_CSV = (
    "processed/splits/test.csv"
)


CHECKPOINT = (
    "checkpoints/scaffold_densenet121_final.pth"
)


RESULTS_DIR = (
    "results"
)


RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "scaffold_densenet121_test_results.csv"
)


PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR,
    "scaffold_densenet121_predictions.csv"
)



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

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")



# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "SCAFFOLD DENSENET-121 TEST EVALUATION"
)

print("=" * 70)


print(
    f"Device: {device}"
)



# ============================================================
# DATASET
# ============================================================

print("\nLoading test dataset...")


test_dataset = BreaKHisDataset(

    TEST_CSV,

    transform=get_eval_transforms(
        IMAGE_SIZE
    )

)



test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS

)



print(
    f"Test samples: {len(test_dataset)}"
)



# ============================================================
# MODEL
# ============================================================

print("\nLoading SCAFFOLD DenseNet-121...")


model = create_densenet121(

    num_classes=2

)



checkpoint = torch.load(

    CHECKPOINT,

    map_location=device

)



if (
    isinstance(checkpoint, dict)
    and
    "model_state_dict" in checkpoint
):

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

else:

    model.load_state_dict(

        checkpoint

    )



model = model.to(device)


model.eval()



print(
    "SCAFFOLD DenseNet-121 loaded successfully."
)



# ============================================================
# EVALUATION
# ============================================================

labels_all = []

predictions_all = []

probabilities_all = []



with torch.no_grad():


    for images, labels in test_loader:


        images = images.to(device)

        labels = labels.to(device)



        outputs = model(images)



        probabilities = torch.softmax(

            outputs,

            dim=1

        )[:,1]



        predictions = torch.argmax(

            outputs,

            dim=1

        )



        labels_all.extend(

            labels.cpu().numpy()

        )


        predictions_all.extend(

            predictions.cpu().numpy()

        )


        probabilities_all.extend(

            probabilities.cpu().numpy()

        )



labels_all = np.array(labels_all)

predictions_all = np.array(predictions_all)

probabilities_all = np.array(probabilities_all)



tn, fp, fn, tp = confusion_matrix(

    labels_all,

    predictions_all

).ravel()



accuracy = accuracy_score(

    labels_all,

    predictions_all

)


precision = precision_score(

    labels_all,

    predictions_all

)


sensitivity = recall_score(

    labels_all,

    predictions_all

)


specificity = tn / (tn + fp)


f1 = f1_score(

    labels_all,

    predictions_all

)


roc_auc = roc_auc_score(

    labels_all,

    probabilities_all

)



print("\n")

print("=" * 70)

print(
    "FINAL SCAFFOLD TEST RESULTS"
)

print("=" * 70)


print(
    "Model: SCAFFOLD DenseNet-121"
)

print(
    f"Accuracy: {accuracy}"
)

print(
    f"Precision: {precision}"
)

print(
    f"Sensitivity: {sensitivity}"
)

print(
    f"Specificity: {specificity}"
)

print(
    f"F1-score: {f1}"
)

print(
    f"ROC-AUC: {roc_auc}"
)


print(
    f"TN: {tn}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print(
    f"TP: {tp}"
)



print("\nClassification Report:")

print(

    classification_report(

        labels_all,

        predictions_all,

        target_names=[

            "Benign",

            "Malignant"

        ]

    )

)



# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(

    [

        {

            "Model":
            "SCAFFOLD DenseNet-121",

            "Accuracy":
            accuracy,

            "Precision":
            precision,

            "Sensitivity":
            sensitivity,

            "Specificity":
            specificity,

            "F1-score":
            f1,

            "ROC-AUC":
            roc_auc,

            "TN":
            tn,

            "FP":
            fp,

            "FN":
            fn,

            "TP":
            tp

        }

    ]

)



results_df.to_csv(

    RESULT_FILE,

    index=False

)



predictions_df = pd.DataFrame(

    {

        "true_label":

        labels_all,


        "prediction":

        predictions_all,


        "probability":

        probabilities_all

    }

)



predictions_df.to_csv(

    PREDICTIONS_FILE,

    index=False

)



print("\nSaved:")

print(
    RESULT_FILE
)

print(
    PREDICTIONS_FILE
)



print("\n")

print("=" * 70)

print(
    "SCAFFOLD EVALUATION COMPLETED"
)

print("=" * 70)