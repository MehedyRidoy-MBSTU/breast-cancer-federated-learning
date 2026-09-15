import os
import random
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.models.densenet121 import create_densenet121
from src.data.transforms import get_train_transforms, get_eval_transforms


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

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
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# CLIENT DATASET
# ============================================================

class ClientCSVDataset(Dataset):

    def __init__(
        self,
        csv_path,
        transform=None
    ):

        self.csv_path = Path(
            csv_path
        )

        if not self.csv_path.exists():

            raise FileNotFoundError(
                f"Client CSV not found: "
                f"{self.csv_path}"
            )

        self.data = pd.read_csv(
            self.csv_path
        )

        required_columns = {
            "path",
            "label"
        }

        missing_columns = (

            required_columns
            -
            set(self.data.columns)

        )

        if missing_columns:

            raise ValueError(
                "Missing columns in "
                f"{self.csv_path}: "
                f"{sorted(missing_columns)}"
            )

        self.transform = transform


    def __len__(self):

        return len(
            self.data
        )


    def __getitem__(
        self,
        index
    ):

        row = self.data.iloc[
            index
        ]

        image_path = str(
            row["path"]
        )

        label = int(
            row["label"]
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# MODEL PARAMETER HELPERS
# ============================================================

def get_model_parameters(
    model
):

    parameters = []

    for tensor in model.state_dict().values():

        parameters.append(

            tensor
            .detach()
            .cpu()
            .numpy()
            .copy()

        )

    return parameters


def set_model_parameters(
    model,
    parameters
):

    state_dict = model.state_dict()

    if len(parameters) != len(state_dict):

        raise ValueError(

            "Model parameter count mismatch. "
            f"Expected {len(state_dict)}, "
            f"received {len(parameters)}."

        )

    new_state_dict = OrderedDict()

    for (
        key,
        current_tensor
    ), array in zip(

        state_dict.items(),
        parameters

    ):

        tensor = torch.as_tensor(
            array,
            dtype=current_tensor.dtype,
            device=current_tensor.device
        )

        if (
            tensor.shape
            !=
            current_tensor.shape
        ):

            raise ValueError(

                f"Shape mismatch for {key}: "
                f"expected "
                f"{tuple(current_tensor.shape)}, "
                f"received "
                f"{tuple(tensor.shape)}"

            )

        new_state_dict[
            key
        ] = tensor

    model.load_state_dict(
        new_state_dict,
        strict=True
    )


# ============================================================
# SCAFFOLD CLIENT
# ============================================================

class ScaffoldClient(
    fl.client.NumPyClient
):

    def __init__(
        self,
        client_id
    ):

        super().__init__()

        self.client_id = str(
            client_id
        )

        print("=" * 70)

        print(
            f"CANONICAL SCAFFOLD CLIENT: "
            f"{self.client_id}"
        )

        print("=" * 70)

        print(
            f"Device: {DEVICE}"
        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = create_densenet121(
            num_classes=2
        )

        self.model = self.model.to(
            DEVICE
        )


        # ----------------------------------------------------
        # PARAMETER COUNTS
        # ----------------------------------------------------

        self.model_parameter_names = list(
            self.model.state_dict().keys()
        )

        self.trainable_parameter_names = [

            name

            for name, _

            in self.model.named_parameters()

        ]

        self.num_model_tensors = len(
            self.model_parameter_names
        )

        self.num_control_tensors = len(
            self.trainable_parameter_names
        )


        # ----------------------------------------------------
        # CLIENT CONTROL VARIATE c_i
        # ----------------------------------------------------

        self.client_control = OrderedDict()

        for (
            name,
            parameter
        ) in self.model.named_parameters():

            self.client_control[
                name
            ] = torch.zeros_like(

                parameter,

                device=DEVICE

            )


        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        csv_path = (

            Path("processed")
            /
            "federated"
            /
            f"{self.client_id}.csv"

        )

        self.train_dataset = ClientCSVDataset(

            csv_path,

            transform=get_train_transforms(
                IMAGE_SIZE
            )

        )

        self.eval_dataset = ClientCSVDataset(

            csv_path,

            transform=get_eval_transforms(
                IMAGE_SIZE
            )

        )


        # Deterministic client-specific shuffle seed

        try:

            numeric_id = int(

                self.client_id
                .split("_")[-1]

            )

        except Exception:

            numeric_id = 0


        loader_generator = (
            torch.Generator()
        )

        loader_generator.manual_seed(

            SEED
            +
            numeric_id

        )


        self.train_loader = DataLoader(

            self.train_dataset,

            batch_size=BATCH_SIZE,

            shuffle=True,

            num_workers=NUM_WORKERS,

            generator=loader_generator

        )


        self.eval_loader = DataLoader(

            self.eval_dataset,

            batch_size=BATCH_SIZE,

            shuffle=False,

            num_workers=NUM_WORKERS

        )


        self.criterion = (
            nn.CrossEntropyLoss()
        )


        print(
            f"Training samples: "
            f"{len(self.train_dataset)}"
        )

        print(
            f"Model tensors: "
            f"{self.num_model_tensors}"
        )

        print(
            f"Control tensors: "
            f"{self.num_control_tensors}"
        )


    # ========================================================
    # GET PARAMETERS
    # ========================================================

    def get_parameters(
        self,
        config
    ):

        # Only model parameters are returned here.
        #
        # The SCAFFOLD server maintains the global
        # control variate separately.

        return get_model_parameters(
            self.model
        )


    # ========================================================
    # SPLIT SERVER FIT PAYLOAD
    # ========================================================

    def _split_server_payload(
        self,
        parameters
    ):

        expected = (

            self.num_model_tensors
            +
            self.num_control_tensors

        )

        if len(parameters) != expected:

            raise ValueError(

                "Invalid SCAFFOLD server payload. "
                f"Expected {expected} arrays "
                f"({self.num_model_tensors} model + "
                f"{self.num_control_tensors} control), "
                f"received {len(parameters)}."

            )


        model_parameters = parameters[
            :
            self.num_model_tensors
        ]


        server_control_arrays = parameters[
            self.num_model_tensors
            :
        ]


        return (
            model_parameters,
            server_control_arrays
        )


    # ========================================================
    # CREATE SERVER CONTROL DICTIONARY
    # ========================================================

    def _create_server_control(
        self,
        control_arrays
    ):

        server_control = OrderedDict()


        for (
            name,
            parameter
        ), array in zip(

            self.model.named_parameters(),
            control_arrays

        ):

            control_tensor = torch.as_tensor(

                array,

                dtype=parameter.dtype,

                device=DEVICE

            )


            if (
                control_tensor.shape
                !=
                parameter.shape
            ):

                raise ValueError(

                    f"Global control shape mismatch "
                    f"for {name}: "
                    f"expected "
                    f"{tuple(parameter.shape)}, "
                    f"received "
                    f"{tuple(control_tensor.shape)}"

                )


            server_control[
                name
            ] = control_tensor


        return server_control


    # ========================================================
    # LOCAL SCAFFOLD TRAINING
    # ========================================================

    def _train_scaffold(
        self,
        server_control
    ):

        self.model.train()


        # ----------------------------------------------------
        # x = global model at start of local training
        # ----------------------------------------------------

        global_trainable = OrderedDict()

        for (
            name,
            parameter
        ) in self.model.named_parameters():

            global_trainable[
                name
            ] = (

                parameter
                .detach()
                .clone()

            )


        # ----------------------------------------------------
        # Canonical SCAFFOLD uses local SGD
        # ----------------------------------------------------

        optimizer = torch.optim.SGD(

            self.model.parameters(),

            lr=LEARNING_RATE,

            momentum=0.0,

            weight_decay=WEIGHT_DECAY

        )


        total_loss = 0.0

        total_samples = 0

        local_steps = 0


        # ----------------------------------------------------
        # Local update:
        #
        # y_i <- y_i -
        # eta * (g_i + c - c_i)
        # ----------------------------------------------------

        for _ in range(
            LOCAL_EPOCHS
        ):

            for (
                images,
                labels
            ) in self.train_loader:

                images = images.to(
                    DEVICE
                )

                labels = labels.to(
                    DEVICE
                )


                optimizer.zero_grad(
                    set_to_none=True
                )


                outputs = self.model(
                    images
                )


                loss = self.criterion(

                    outputs,

                    labels

                )


                loss.backward()


                with torch.no_grad():

                    for (
                        name,
                        parameter
                    ) in self.model.named_parameters():

                        if (
                            parameter.grad
                            is None
                        ):

                            continue


                        correction = (

                            server_control[
                                name
                            ]

                            -

                            self.client_control[
                                name
                            ]

                        )


                        parameter.grad.add_(
                            correction
                        )


                optimizer.step()


                batch_size = (
                    images.size(0)
                )


                total_loss += (

                    loss.item()
                    *
                    batch_size

                )


                total_samples += (
                    batch_size
                )


                local_steps += 1


        if local_steps == 0:

            raise RuntimeError(
                "No local SCAFFOLD "
                "training steps were executed."
            )


        # ----------------------------------------------------
        # OPTION II CONTROL UPDATE
        #
        # c_i+ =
        # c_i - c +
        # (x - y_i) / (K * eta_l)
        #
        # Δc_i = c_i+ - c_i
        # ----------------------------------------------------

        control_delta_arrays = []

        control_delta_norm_sq = 0.0


        denominator = (

            local_steps
            *
            LEARNING_RATE

        )


        with torch.no_grad():

            for (
                name,
                parameter
            ) in self.model.named_parameters():

                old_client_control = (

                    self.client_control[
                        name
                    ]

                )


                new_client_control = (

                    old_client_control

                    -

                    server_control[
                        name
                    ]

                    +

                    (
                        global_trainable[
                            name
                        ]

                        -

                        parameter.detach()

                    )
                    /
                    denominator

                )


                control_delta = (

                    new_client_control

                    -

                    old_client_control

                )


                control_delta_norm_sq += float(

                    torch.sum(

                        control_delta
                        *
                        control_delta

                    )
                    .detach()
                    .cpu()
                    .item()

                )


                self.client_control[
                    name
                ] = (

                    new_client_control
                    .detach()
                    .clone()

                )


                control_delta_arrays.append(

                    control_delta
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()

                )


        average_loss = (

            total_loss
            /
            max(
                total_samples,
                1
            )

        )


        control_delta_l2 = float(

            np.sqrt(
                control_delta_norm_sq
            )

        )


        return (

            average_loss,

            local_steps,

            control_delta_l2,

            control_delta_arrays

        )


    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        parameters,
        config
    ):

        print("\n" + "-" * 70)

        print(
            f"{self.client_id}: "
            "SCAFFOLD FIT"
        )

        print("-" * 70)


        (
            model_parameters,
            server_control_arrays

        ) = self._split_server_payload(
            parameters
        )


        # Load global model x

        set_model_parameters(

            self.model,

            model_parameters

        )


        # Load global control c

        server_control = (
            self._create_server_control(
                server_control_arrays
            )
        )


        # Train locally

        (
            train_loss,
            local_steps,
            control_delta_l2,
            control_delta_arrays

        ) = self._train_scaffold(
            server_control
        )


        # Updated local model y_i

        local_model_parameters = (
            get_model_parameters(
                self.model
            )
        )


        # Client sends:
        #
        # [updated model y_i]
        # +
        # [Δc_i]

        outgoing_parameters = (

            local_model_parameters

            +

            control_delta_arrays

        )


        print(
            f"{self.client_id}: "
            f"train_loss="
            f"{train_loss:.6f}"
        )

        print(
            f"{self.client_id}: "
            f"local_steps="
            f"{local_steps}"
        )

        print(
            f"{self.client_id}: "
            f"control_delta_l2="
            f"{control_delta_l2:.6f}"
        )


        return (

            outgoing_parameters,

            len(
                self.train_dataset
            ),

            {

                "train_loss":
                    float(
                        train_loss
                    ),

                "local_steps":
                    int(
                        local_steps
                    ),

                "control_delta_l2":
                    float(
                        control_delta_l2
                    )

            }

        )


    # ========================================================
    # EVALUATION
    # ========================================================

    def evaluate(
        self,
        parameters,
        config
    ):

        # Evaluation receives only the
        # current global MODEL parameters,
        # not SCAFFOLD control variates.

        if (
            len(parameters)
            !=
            self.num_model_tensors
        ):

            raise ValueError(

                "Invalid evaluation payload. "
                f"Expected "
                f"{self.num_model_tensors} "
                f"model tensors, "
                f"received "
                f"{len(parameters)}."

            )


        set_model_parameters(

            self.model,

            parameters

        )


        self.model.eval()


        total_loss = 0.0

        total_correct = 0

        total_samples = 0


        with torch.no_grad():

            for (
                images,
                labels
            ) in self.eval_loader:

                images = images.to(
                    DEVICE
                )

                labels = labels.to(
                    DEVICE
                )


                outputs = self.model(
                    images
                )


                loss = self.criterion(

                    outputs,

                    labels

                )


                predictions = torch.argmax(

                    outputs,

                    dim=1

                )


                batch_size = (
                    images.size(0)
                )


                total_loss += (

                    loss.item()
                    *
                    batch_size

                )


                total_correct += int(

                    (
                        predictions
                        ==
                        labels
                    )
                    .sum()
                    .item()

                )


                total_samples += (
                    batch_size
                )


        average_loss = (

            total_loss
            /
            max(
                total_samples,
                1
            )

        )


        accuracy = (

            total_correct
            /
            max(
                total_samples,
                1
            )

        )


        return (

            float(
                average_loss
            ),

            int(
                total_samples
            ),

            {

                "accuracy":
                    float(
                        accuracy
                    )

            }

        )