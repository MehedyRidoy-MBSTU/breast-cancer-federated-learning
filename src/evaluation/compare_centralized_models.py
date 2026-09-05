import pandas as pd
import os


RESULT_DIR = "results"


# ------------------------------------------------------------
# Load prediction files
# ------------------------------------------------------------

resnet = pd.read_csv(
    os.path.join(
        RESULT_DIR,
        "resnet18_test_predictions.csv"
    )
)

dense = pd.read_csv(
    os.path.join(
        RESULT_DIR,
        "densenet121_test_predictions.csv"
    )
)


# ------------------------------------------------------------
# Metric calculator
# ------------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def calculate_metrics(df, threshold):

    y_true = df["true_label"].values

    y_prob = df[
        "malignant_probability"
    ].values


    y_pred = (
        y_prob >= threshold
    ).astype(int)


    acc = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )


    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()


    specificity = (
        tn /
        (tn + fp)
    )


    return [
        acc,
        precision,
        recall,
        specificity,
        f1,
        auc
    ]



# ------------------------------------------------------------
# Create comparison
# ------------------------------------------------------------

results = []


models = [
    (
        "ResNet-18",
        resnet
    ),
    (
        "DenseNet-121",
        dense
    )
]


thresholds = {
    "ResNet-18": [
        0.50,
        0.10
    ],

    "DenseNet-121": [
        0.50,
        0.07
    ]
}



for name, data in models:

    for threshold in thresholds[name]:

        metrics = calculate_metrics(
            data,
            threshold
        )

        results.append(
            [
                name,
                threshold,
                *metrics
            ]
        )


comparison = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Threshold",
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1-score",
        "ROC-AUC"
    ]
)



output = os.path.join(
    RESULT_DIR,
    "centralized_model_comparison.csv"
)


comparison.to_csv(
    output,
    index=False
)


print("="*70)
print("CENTRALIZED MODEL COMPARISON")
print("="*70)

print(
    comparison.to_string(
        index=False
    )
)


print("\nSaved:")
print(output)