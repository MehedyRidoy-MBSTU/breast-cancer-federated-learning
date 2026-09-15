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
    f1_score,
    roc_auc_score,
)

from src.models.densenet121 import create_densenet121

from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
)

from src.federated.datasets.federated_dataset import (
    FederatedBreaKHisDataset,
)

from src.privacy.dp_utils import (
    dp_gradient_step
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
# DIFFERENTIAL PRIVACY CONFIGURATION
# ============================================================

DP_MAX_GRAD_NORM = 1.0

DP_NOISE_MULTIPLIER = 1.0



# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cpu"
)



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



def set_model_parameters(
    model,
    parameters
):

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



# ============================================================
# DATA LOADERS
# ============================================================

def create_client_dataloaders(
    client_id
):

    csv_path = (

        Path("processed")
        /
        "federated"
        /
        f"{client_id}.csv"

    )


    train_dataset = FederatedBreaKHisDataset(

        csv_file=csv_path,

        transform=get_train_transforms(
            IMAGE_SIZE
        )

    )


    eval_dataset = FederatedBreaKHisDataset(

        csv_file=csv_path,

        transform=get_eval_transforms(
            IMAGE_SIZE
        )

    )


    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS

    )


    eval_loader = torch.utils.data.DataLoader(

        eval_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS

    )


    return (

        train_dataset,

        eval_dataset,

        train_loader,

        eval_loader

    )



# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights_from_csv(
    client_id
):

    csv_path = (

        Path("processed")

        /

        "federated"

        /

        f"{client_id}.csv"

    )


    df = pd.read_csv(
        csv_path
    )


    labels = df["label"].values


    class_counts = np.bincount(
        labels,
        minlength=2
    )


    total = len(labels)


    weights = torch.tensor(

        [

            total /
            (2 * class_counts[0]),


            total /
            (2 * class_counts[1])

        ],

        dtype=torch.float32,

        device=DEVICE

    )


    return weights
# ============================================================
# LOCAL TRAINING WITH DIFFERENTIAL PRIVACY
# ============================================================

def train_local_model(
    model,
    train_loader,
    criterion,
    optimizer,
    local_epochs=1,
):

    model.train()

    epoch_losses = []


    for epoch in range(local_epochs):

        running_loss = 0.0

        total_samples = 0


        for images, labels in train_loader:


            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            optimizer.zero_grad()


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()


            # ====================================================
            # DP-SGD STEP
            # Gradient clipping + Gaussian noise
            # ====================================================

            dp_gradient_step(

                model,

                optimizer,

                max_norm=DP_MAX_GRAD_NORM,

                noise_multiplier=DP_NOISE_MULTIPLIER

            )


            running_loss += (

                loss.item()

                *

                labels.size(0)

            )


            total_samples += labels.size(0)



        epoch_loss = (

            running_loss

            /

            total_samples

        )


        epoch_losses.append(
            epoch_loss
        )


    return float(
        np.mean(epoch_losses)
    )



# ============================================================
# LOCAL EVALUATION
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


            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


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

                labels.cpu()
                .numpy()

            )


            preds_all.extend(

                preds.cpu()
                .numpy()

            )


            probs_all.extend(

                probs.cpu()
                .numpy()

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



    metrics = {

        "accuracy":

        accuracy_score(
            labels_all,
            preds_all
        ),


        "f1":

        f1_score(
            labels_all,
            preds_all
        ),


        "roc_auc":

        roc_auc_score(
            labels_all,
            probs_all
        )

    }


    return (

        total_loss /
        len(eval_loader.dataset),

        metrics

    )



# ============================================================
# FLOWER CLIENT
# ============================================================

class DP_DenseNetClient(
    fl.client.NumPyClient
):


    def __init__(
        self,
        client_id
    ):


        self.client_id = client_id


        self.model = create_densenet121(

            num_classes=2

        ).to(
            DEVICE
        )


        (

            self.train_dataset,

            self.eval_dataset,

            self.train_loader,

            self.eval_loader

        ) = create_client_dataloaders(
            client_id
        )



        self.class_weights = (

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

            weight_decay=WEIGHT_DECAY

        )



    def get_parameters(
        self,
        config
    ):

        return get_model_parameters(
            self.model
        )



    def fit(
        self,
        parameters,
        config
    ):


        set_model_parameters(

            self.model,

            parameters

        )


        local_epochs = int(

            config.get(
                "local_epochs",
                1
            )

        )


        train_loss = train_local_model(

            self.model,

            self.train_loader,

            self.criterion,

            self.optimizer,

            local_epochs

        )



        return (

            get_model_parameters(
                self.model
            ),

            len(
                self.train_dataset
            ),

            {

                "client_id":

                self.client_id,


                "train_loss":

                train_loss,


                "dp_noise_multiplier":

                DP_NOISE_MULTIPLIER,


                "dp_max_grad_norm":

                DP_MAX_GRAD_NORM

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


        loss, metrics = evaluate_local_model(

            self.model,

            self.eval_loader,

            self.criterion

        )


        return (

            float(loss),

            len(
                self.eval_dataset
            ),

            metrics

        )



# ============================================================
# CLIENT FACTORY
# ============================================================

def create_flower_client(
    client_id
):

    return DP_DenseNetClient(
        client_id
    ).to_client()