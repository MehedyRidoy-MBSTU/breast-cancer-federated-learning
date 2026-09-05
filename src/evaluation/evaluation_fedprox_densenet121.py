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
    "checkpoints/fedprox_densenet121_final.pth"
)


RESULTS_DIR = (
    "results"
)


RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "fedprox_densenet121_test_results.csv"
)


PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR,
    "fedprox_densenet121_predictions.csv"
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
    "FEDPROX DENSENET-121 TEST EVALUATION"
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

print("\nLoading FedProx DenseNet-121...")


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
    "FedProx DenseNet-121 loaded successfully."
)



# ============================================================
# EVALUATION
# ============================================================

all_labels = []

all_predictions = []

all_probabilities = []



criterion = torch.nn.CrossEntropyLoss()


total_loss = 0.0



with torch.no_grad():


    for images, labels in test_loader:


        images = images.to(device)

        labels = labels.to(device)



        outputs = model(images)



        loss = criterion(

            outputs,

            labels

        )



        probabilities = torch.softmax(

            outputs,

            dim=1

        )



        predictions = torch.argmax(

            outputs,

            dim=1

        )



        total_loss += (

            loss.item()

            *

            labels.size(0)

        )



        all_labels.extend(

            labels.cpu().numpy()

        )


        all_predictions.extend(

            predictions.cpu().numpy()

        )


        all_probabilities.extend(

            probabilities[:,1]
            .cpu()
            .numpy()

        )



# ============================================================
# NUMPY CONVERSION
# ============================================================

all_labels = np.array(
    all_labels
)


all_predictions = np.array(
    all_predictions
)


all_probabilities = np.array(
    all_probabilities
)



test_loss = (

    total_loss

    /

    len(test_dataset)

)



# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(

    all_labels,

    all_predictions

)


precision = precision_score(

    all_labels,

    all_predictions,

    zero_division=0

)



recall = recall_score(

    all_labels,

    all_predictions,

    zero_division=0

)



f1 = f1_score(

    all_labels,

    all_predictions,

    zero_division=0

)



roc_auc = roc_auc_score(

    all_labels,

    all_probabilities

)



# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    all_predictions

)



tn, fp, fn, tp = cm.ravel()



specificity = (

    tn /

    (tn + fp)

)



# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")

print("=" * 70)

print(
    "FINAL FEDPROX TEST RESULTS"
)

print("=" * 70)



print(
    f"Model: FedProx DenseNet-121"
)


print(
    f"Accuracy: {accuracy}"
)


print(
    f"Precision: {precision}"
)


print(
    f"Sensitivity: {recall}"
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

        all_labels,

        all_predictions,

        target_names=[
            "Benign",
            "Malignant"
        ],

        digits=2

    )

)



# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(

    RESULTS_DIR,

    exist_ok=True

)



results = pd.DataFrame(

    [

        {

            "Model":
            "FedProx DenseNet-121",

            "Accuracy":
            accuracy,

            "Precision":
            precision,

            "Sensitivity":
            recall,

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
            tp,

        }

    ]

)



results.to_csv(

    RESULT_FILE,

    index=False

)



prediction_df = pd.DataFrame(

    {

        "true_label":
        all_labels,

        "prediction":
        all_predictions,

        "probability":
        all_probabilities

    }

)



prediction_df.to_csv(

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
    "FEDPROX EVALUATION COMPLETED"
)

print("=" * 70)