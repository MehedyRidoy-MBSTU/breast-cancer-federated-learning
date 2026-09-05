import warnings
warnings.filterwarnings("ignore")

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import flwr as fl

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.models.densenet121 import create_densenet121

from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
)

from src.federated.datasets.federated_dataset import (
    FederatedBreaKHisDataset,
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# PARAMETER UTILITIES
# ============================================================

def get_model_parameters(model):

    return [
        value.detach()
        .cpu()
        .numpy()
        for _, value in model.state_dict().items()
    ]



def set_model_parameters(model, parameters):

    print("DEBUG: set_model_parameters started")

    params_dict = zip(
        model.state_dict().keys(),
        parameters
    )

    state_dict = OrderedDict(
        {
            key: torch.tensor(value)
            for key, value in params_dict
        }
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print("DEBUG: set_model_parameters completed")



# ============================================================
# DATA LOADER
# ============================================================

def create_client_dataloaders(
    client_id,
):

    csv_path = (
        Path("processed")
        / "federated"
        / f"{client_id}.csv"
    )


    train_dataset = FederatedBreaKHisDataset(
        csv_file=csv_path,
        transform=get_train_transforms(IMAGE_SIZE),
    )


    eval_dataset = FederatedBreaKHisDataset(
        csv_file=csv_path,
        transform=get_eval_transforms(IMAGE_SIZE),
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )


    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


    return (
        train_dataset,
        eval_dataset,
        train_loader,
        eval_loader,
    )



# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights_from_csv(client_id):

    csv_path = (
        Path("processed")
        / "federated"
        / f"{client_id}.csv"
    )


    df = pd.read_csv(csv_path)

    labels = df["label"].values

    class_counts = np.bincount(
        labels,
        minlength=2
    )

    total = len(labels)


    weights = torch.tensor(
        [
            total / (2 * class_counts[0]),
            total / (2 * class_counts[1]),
        ],
        dtype=torch.float32,
        device=DEVICE,
    )


    return weights, class_counts



# ============================================================
# LOCAL TRAINING
# ============================================================

def train_local_model(
    model,
    train_loader,
    criterion,
    optimizer,
    local_epochs=1,
):

    print("DEBUG: train_local_model started")

    model.train()

    total_batches = len(train_loader)

    print(
        f"DEBUG: Total batches: {total_batches}"
    )


    epoch_losses = []


    for epoch in range(local_epochs):

        print(
            f"DEBUG: Starting epoch {epoch+1}"
        )


        running_loss = 0.0
        total_samples = 0


        for batch_idx, (images, labels) in enumerate(train_loader):

            if batch_idx == 0:
                print(
                    "DEBUG: First batch loaded"
                )


            images = images.to(DEVICE)
            labels = labels.to(DEVICE)


            optimizer.zero_grad()


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()


            optimizer.step()


            running_loss += (
                loss.item()
                *
                labels.size(0)
            )

            total_samples += labels.size(0)


            if batch_idx % 20 == 0:
                print(
                    f"DEBUG: batch {batch_idx}/{total_batches}"
                )


        epoch_loss = (
            running_loss /
            total_samples
        )


        epoch_losses.append(epoch_loss)


        print(
            f"DEBUG: Epoch completed loss={epoch_loss}"
        )


    print("DEBUG: train_local_model completed")


    return float(
        np.mean(epoch_losses)
    )



# ============================================================
# EVALUATION
# ============================================================

def evaluate_local_model(
    model,
    eval_loader,
    criterion,
):

    model.eval()

    total_loss = 0

    labels_all = []
    preds_all = []
    probs_all = []


    with torch.no_grad():

        for images, labels in eval_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            probs = torch.softmax(
                outputs,
                dim=1
            )[:,1]


            preds = torch.argmax(
                outputs,
                dim=1
            )


            total_loss += (
                loss.item()
                *
                labels.size(0)
            )


            labels_all.extend(
                labels.cpu().numpy()
            )

            preds_all.extend(
                preds.cpu().numpy()
            )

            probs_all.extend(
                probs.cpu().numpy()
            )



    labels_all=np.array(labels_all)
    preds_all=np.array(preds_all)
    probs_all=np.array(probs_all)


    acc=accuracy_score(
        labels_all,
        preds_all
    )

    f1=f1_score(
        labels_all,
        preds_all
    )


    auc=roc_auc_score(
        labels_all,
        probs_all
    )


    return (
        total_loss/len(eval_loader.dataset),
        {
            "accuracy":acc,
            "f1":f1,
            "roc_auc":auc,
        }
    )



# ============================================================
# FLOWER CLIENT
# ============================================================

class DenseNetClient(fl.client.NumPyClient):


    def __init__(self, client_id):

        self.client_id = client_id


        print(
            "DEBUG: Creating DenseNet model"
        )


        self.model = create_densenet121(
            num_classes=2
        ).to(DEVICE)


        (
            self.train_dataset,
            self.eval_dataset,
            self.train_loader,
            self.eval_loader,

        ) = create_client_dataloaders(
            client_id
        )


        self.class_weights, self.class_counts = (
            compute_class_weights_from_csv(
                client_id
            )
        )


        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights
        )


        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )



    def get_parameters(self, config):

        return get_model_parameters(
            self.model
        )



    def fit(
        self,
        parameters,
        config
    ):


        print("\n========== FIT START ==========")


        print(
            "DEBUG 1: Loading global parameters"
        )


        set_model_parameters(
            self.model,
            parameters
        )


        print(
            "DEBUG 2: Parameters loaded"
        )


        local_epochs = int(
            config.get(
                "local_epochs",
                1
            )
        )


        print(
            f"DEBUG 3: Starting training epochs={local_epochs}"
        )


        train_loss = train_local_model(
            self.model,
            self.train_loader,
            self.criterion,
            self.optimizer,
            local_epochs,
        )


        print(
            "DEBUG 4: Training finished"
        )


        return (
            get_model_parameters(self.model),
            len(self.train_dataset),
            {
                "client_id":self.client_id,
                "train_loss":train_loss,
            }
        )



    def evaluate(
        self,
        parameters,
        config
    ):

        set_model_parameters(
            self.model,
            parameters
        )


        loss,metrics = evaluate_local_model(
            self.model,
            self.eval_loader,
            self.criterion,
        )


        return (
            float(loss),
            len(self.eval_dataset),
            metrics,
        )



def create_flower_client(client_id):

    return DenseNetClient(
        client_id
    ).to_client()