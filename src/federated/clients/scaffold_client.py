import warnings
warnings.filterwarnings("ignore")


from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import flwr as fl


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

LOCAL_EPOCHS = 1

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4



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
# PARAMETER UTILITIES
# ============================================================

def get_parameters(model):

    return [

        value.detach()
        .cpu()
        .numpy()

        for _, value in model.state_dict().items()

    ]



def set_parameters(model, parameters):

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
# DATA
# ============================================================

def create_client_dataloaders(client_id):


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
# SCAFFOLD TRAINING
# ============================================================

def train_scaffold(

    model,

    global_control,

    client_control,

    train_loader,

    criterion,

    optimizer

):


    model.train()


    old_parameters = {

        name:

        param.detach().clone()

        for name, param in model.named_parameters()

    }


    for images, labels in train_loader:


        images = images.to(DEVICE)

        labels = labels.to(DEVICE)



        optimizer.zero_grad()



        outputs = model(images)


        loss = criterion(

            outputs,

            labels

        )



        loss.backward()



        # SCAFFOLD correction

        with torch.no_grad():

            for name, param in model.named_parameters():


                if param.grad is not None:


                    param.grad += (

                        global_control[name]

                        -

                        client_control[name]

                    )



        optimizer.step()



    # update client control variate

    new_control = {}


    for name, param in model.named_parameters():


        new_control[name] = (

            client_control[name]

            -

            (param.detach() - old_parameters[name])

            /

            (LOCAL_EPOCHS * LEARNING_RATE)

        )


    return new_control



# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(

    model,

    loader,

    criterion

):

    model.eval()


    total_loss = 0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in loader:


            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            loss = criterion(

                outputs,

                labels

            )


            total_loss += loss.item()



            predictions = torch.argmax(

                outputs,

                dim=1

            )


            correct += (

                predictions == labels

            ).sum().item()


            total += labels.size(0)



    return (

        total_loss / len(loader),

        correct / total

    )



# ============================================================
# SCAFFOLD CLIENT
# ============================================================

class ScaffoldClient(fl.client.NumPyClient):


    def __init__(self, client_id):


        self.client_id = client_id


        self.model = create_densenet121(

            num_classes=2

        ).to(DEVICE)



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



        self.client_control = {

            name:

            torch.zeros_like(param)

            for name, param in self.model.named_parameters()

        }



    def get_parameters(self, config):


        return get_parameters(

            self.model

        )



    def fit(self, parameters, config):


        set_parameters(

            self.model,

            parameters

        )


        global_control = {

            name:

            torch.zeros_like(param)

            for name, param in self.model.named_parameters()

        }


        self.client_control = train_scaffold(

            self.model,

            global_control,

            self.client_control,

            self.train_loader,

            self.criterion,

            self.optimizer

        )


        return (

            get_parameters(

                self.model

            ),

            len(self.train_dataset),

            {}

        )



    def evaluate(self, parameters, config):


        set_parameters(

            self.model,

            parameters

        )


        loss, accuracy = evaluate_model(

            self.model,

            self.eval_loader,

            self.criterion

        )


        return (

            float(loss),

            len(self.eval_dataset),

            {

                "accuracy":

                accuracy

            }

        )