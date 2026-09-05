import warnings
warnings.filterwarnings("ignore")


from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import flwr as fl


from sklearn.metrics import (
    accuracy_score,
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


PROXIMAL_MU = 0.0001



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
# DATA LOADER
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
# FEDPROX TRAINING
# ============================================================


def train_local_model(

    model,

    global_parameters,

    train_loader,

    criterion,

    optimizer,

    epochs

):


    model.train()


    total_loss = 0.0



    # Correct mapping between global and local weights

    global_state = OrderedDict(

        {

            key: torch.tensor(

                value,

                device=DEVICE

            )

            for key, value in zip(

                model.state_dict().keys(),

                global_parameters

            )

        }

    )



    for epoch in range(epochs):


        for images, labels in train_loader:


            images = images.to(
                DEVICE
            )


            labels = labels.to(
                DEVICE
            )



            optimizer.zero_grad()



            outputs = model(images)



            ce_loss = criterion(

                outputs,

                labels

            )



            proximal_loss = torch.tensor(

                0.0,

                device=DEVICE

            )



            for name, local_param in model.state_dict().items():


                global_param = global_state[name]


                proximal_loss += torch.sum(

                    torch.pow(

                        local_param - global_param,

                        2

                    )

                )



            loss = (

                ce_loss

                +

                (PROXIMAL_MU / 2)

                *

                proximal_loss

            )



            loss.backward()


            optimizer.step()



            total_loss += loss.item()



    return total_loss / len(train_loader)



# ============================================================
# EVALUATION
# ============================================================


def evaluate_local_model(

    model,

    loader,

    criterion

):


    model.eval()


    labels_all = []

    preds_all = []

    probs_all = []


    total_loss = 0.0



    with torch.no_grad():


        for images, labels in loader:


            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(images)


            loss = criterion(

                outputs,

                labels

            )


            total_loss += loss.item()



            probs = torch.softmax(

                outputs,

                dim=1

            )[:,1]



            preds = torch.argmax(

                outputs,

                dim=1

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



    labels_all = np.array(labels_all)

    preds_all = np.array(preds_all)

    probs_all = np.array(probs_all)



    tn, fp, fn, tp = confusion_matrix(

        labels_all,

        preds_all

    ).ravel()



    return (

        total_loss / len(loader),

        {

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

    )



# ============================================================
# FEDPROX CLIENT
# ============================================================


class FedProxClient(
    fl.client.NumPyClient
):


    def __init__(self, client_id):


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



        self.criterion = nn.CrossEntropyLoss()



        self.optimizer = torch.optim.AdamW(

            self.model.parameters(),

            lr=LEARNING_RATE,

            weight_decay=WEIGHT_DECAY

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


        print(
            "\n========== FEDPROX FIT START =========="
        )


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


        loss = train_local_model(

            self.model,

            parameters,

            self.train_loader,

            self.criterion,

            self.optimizer,

            local_epochs

        )


        print(
            "FedProx training completed"
        )


        return (

            get_model_parameters(

                self.model

            ),

            len(self.train_dataset),

            {

                "client_id":

                self.client_id,

                "train_loss":

                loss

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

            len(self.eval_dataset),

            metrics

        )