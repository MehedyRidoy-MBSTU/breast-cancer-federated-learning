import warnings
warnings.filterwarnings("ignore")


from pathlib import Path

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


from src.models.densenet121 import create_densenet121

from src.data.transforms import get_eval_transforms

from src.federated.datasets.federated_dataset import (
    FederatedBreaKHisDataset
)



# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0



MODEL_PATH = (
    "checkpoints/"
    "dp_fedavg_densenet121_final.pth"
)



TEST_CSV = (
    Path("processed")
    /
    "splits"
    /
    "test.csv"
)



RESULT_DIR = Path(
    "results"
)

RESULT_DIR.mkdir(
    exist_ok=True
)



RESULT_FILE = (

    RESULT_DIR
    /
    "dp_fedavg_densenet121_test_results.csv"

)



PREDICTION_FILE = (

    RESULT_DIR
    /
    "dp_fedavg_densenet121_predictions.csv"

)



# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device(
        "mps"
    )

elif torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )



# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\nLoading DP-FedAvg DenseNet-121..."
    )


    model = create_densenet121(
        num_classes=2
    )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )


    model.load_state_dict(
        checkpoint
    )


    model.to(
        DEVICE
    )


    model.eval()


    print(
        "DP-FedAvg model loaded successfully."
    )


    return model



# ============================================================
# DATASET
# ============================================================

def create_test_loader():


    print(
        "\nLoading test dataset..."
    )


    dataset = FederatedBreaKHisDataset(

        csv_file=TEST_CSV,

        transform=get_eval_transforms(
            IMAGE_SIZE
        )

    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS

    )


    print(
        f"Test samples: {len(dataset)}"
    )


    return loader



# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader
):


    labels_all = []

    preds_all = []

    probs_all = []



    with torch.no_grad():


        for images, labels in loader:


            images = images.to(
                DEVICE
            )


            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


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


            preds_all.extend(
                predictions.cpu().numpy()
            )


            probs_all.extend(
                probabilities.cpu().numpy()
            )



    labels_all = np.array(
        labels_all
    )


    preds_all = np.array(
        preds_all
    )


    probs_all = np.array(
        probs_all
    )



    accuracy = accuracy_score(
        labels_all,
        preds_all
    )


    precision = precision_score(
        labels_all,
        preds_all,
        zero_division=0
    )


    sensitivity = recall_score(
        labels_all,
        preds_all,
        zero_division=0
    )


    f1 = f1_score(
        labels_all,
        preds_all,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        labels_all,
        probs_all
    )



    cm = confusion_matrix(
        labels_all,
        preds_all
    )


    tn, fp, fn, tp = cm.ravel()



    specificity = (

        tn /
        (tn + fp)

        if (tn + fp) > 0

        else 0

    )



    results = {


        "Model":

        "DP-FedAvg DenseNet-121",



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

        tp,

    }



    predictions = pd.DataFrame(

        {

            "true_label":
            labels_all,


            "prediction":
            preds_all,


            "probability":
            probs_all,

        }

    )



    return results, predictions



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    print("=" * 70)

    print(
        "DP-FEDAVG DENSENET-121 TEST EVALUATION"
    )

    print("=" * 70)


    print(
        f"Device: {DEVICE}"
    )


    model = load_model()


    loader = create_test_loader()



    results, predictions = evaluate(

        model,

        loader

    )



    print("\n")

    print("=" * 70)

    print(
        "FINAL DP-FEDAVG TEST RESULTS"
    )

    print("=" * 70)



    for key, value in results.items():

        print(
            f"{key}: {value}"
        )



    print("\nClassification Report:")


    print(

        classification_report(

            predictions["true_label"],

            predictions["prediction"],

            target_names=[

                "Benign",

                "Malignant"

            ]

        )

    )



    pd.DataFrame(

        [results]

    ).to_csv(

        RESULT_FILE,

        index=False

    )



    predictions.to_csv(

        PREDICTION_FILE,

        index=False

    )



    print("\nSaved:")

    print(
        RESULT_FILE
    )

    print(
        PREDICTION_FILE
    )


    print("\n")

    print("=" * 70)

    print(
        "DP-FEDAVG EVALUATION COMPLETED"
    )

    print("=" * 70)