import random
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import numpy as np
import torch

from flwr.common import (
    EvaluateIns,
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from src.models.densenet121 import create_densenet121


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

SERVER_ADDRESS = "127.0.0.1:8080"

NUM_ROUNDS = 5

NUM_CLIENTS = 5

SERVER_LEARNING_RATE = 1.0

CHECKPOINT_DIR = Path(
    "checkpoints"
)

FINAL_MODEL_PATH = (
    CHECKPOINT_DIR
    /
    "scaffold_densenet121_final.pth"
)

FINAL_CONTROL_PATH = (
    CHECKPOINT_DIR
    /
    "scaffold_densenet121_final_control.pth"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL HELPERS
# ============================================================

def get_model_ndarrays(
    model
):

    return [

        tensor
        .detach()
        .cpu()
        .numpy()
        .copy()

        for tensor
        in model.state_dict().values()

    ]


def load_ndarrays_into_model(
    model,
    arrays
):

    state_dict = model.state_dict()

    if len(arrays) != len(state_dict):

        raise ValueError(

            "Model tensor count mismatch. "
            f"Expected {len(state_dict)}, "
            f"received {len(arrays)}."

        )


    new_state_dict = OrderedDict()


    for (
        key,
        reference_tensor
    ), array in zip(

        state_dict.items(),
        arrays

    ):

        tensor = torch.as_tensor(

            array,

            dtype=reference_tensor.dtype

        )


        if (
            tensor.shape
            !=
            reference_tensor.shape
        ):

            raise ValueError(

                f"Shape mismatch for {key}: "
                f"expected "
                f"{tuple(reference_tensor.shape)}, "
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


def save_model_checkpoint(
    model_arrays,
    output_path
):

    model = create_densenet121(
        num_classes=2
    )


    load_ndarrays_into_model(

        model,

        model_arrays

    )


    torch.save(

        model.state_dict(),

        output_path

    )


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def mean_arrays(
    arrays
):

    if len(arrays) == 0:

        raise ValueError(
            "Cannot average an empty array list."
        )


    reference = arrays[0]


    # Floating tensors
    if np.issubdtype(
        reference.dtype,
        np.floating
    ):

        accumulator = np.zeros_like(

            reference,

            dtype=np.float64

        )


        for array in arrays:

            accumulator += array.astype(

                np.float64,

                copy=False

            )


        accumulator /= len(
            arrays
        )


        return accumulator.astype(

            reference.dtype,

            copy=False

        )


    # Integer state tensors such as
    # BatchNorm num_batches_tracked.
    #
    # Average in float64 and round back
    # to the original integer dtype.

    accumulator = np.zeros(

        reference.shape,

        dtype=np.float64

    )


    for array in arrays:

        accumulator += array.astype(

            np.float64,

            copy=False

        )


    accumulator /= len(
        arrays
    )


    accumulator = np.rint(
        accumulator
    )


    return accumulator.astype(

        reference.dtype

    )


def l2_norm_of_arrays(
    arrays
):

    squared_norm = 0.0


    for array in arrays:

        array64 = array.astype(

            np.float64,

            copy=False

        )


        squared_norm += float(

            np.sum(
                array64 * array64
            )

        )


    return float(
        np.sqrt(
            squared_norm
        )
    )


# ============================================================
# METRIC AGGREGATION
# ============================================================

def aggregate_fit_metrics(
    metrics
):

    if not metrics:

        return {}


    total_examples = sum(

        num_examples

        for num_examples, _
        in metrics

    )


    if total_examples == 0:

        return {}


    aggregated = {}


    metric_names = set()


    for _, metric_dict in metrics:

        metric_names.update(
            metric_dict.keys()
        )


    for name in metric_names:

        weighted_sum = 0.0

        used_examples = 0


        for (
            num_examples,
            metric_dict
        ) in metrics:

            if name not in metric_dict:

                continue


            value = metric_dict[
                name
            ]


            if isinstance(
                value,
                (
                    int,
                    float
                )
            ):

                weighted_sum += (

                    num_examples
                    *
                    float(value)

                )


                used_examples += (
                    num_examples
                )


        if used_examples > 0:

            aggregated[
                name
            ] = (

                weighted_sum
                /
                used_examples

            )


    return aggregated


def aggregate_evaluate_metrics(
    metrics
):

    return aggregate_fit_metrics(
        metrics
    )


# ============================================================
# CANONICAL SCAFFOLD STRATEGY
# ============================================================

class ScaffoldStrategy(
    fl.server.strategy.FedAvg
):

    def __init__(
        self,
        initial_model_arrays,
        initial_control_arrays,
        num_model_tensors,
        num_control_tensors
    ):

        self.num_model_tensors = int(
            num_model_tensors
        )

        self.num_control_tensors = int(
            num_control_tensors
        )


        # ----------------------------------------------------
        # Current global model x
        # ----------------------------------------------------

        self.current_global_model = [

            array.copy()

            for array
            in initial_model_arrays

        ]


        # ----------------------------------------------------
        # Global SCAFFOLD control variate c
        # ----------------------------------------------------

        self.global_control = [

            array.copy()

            for array
            in initial_control_arrays

        ]


        # ----------------------------------------------------
        # Latest successfully aggregated model
        # ----------------------------------------------------

        self.latest_model = [

            array.copy()

            for array
            in initial_model_arrays

        ]


        super().__init__(

            fraction_fit=1.0,

            fraction_evaluate=1.0,

            min_fit_clients=NUM_CLIENTS,

            min_evaluate_clients=NUM_CLIENTS,

            min_available_clients=NUM_CLIENTS,

            accept_failures=False,

            initial_parameters=(
                ndarrays_to_parameters(
                    initial_model_arrays
                )
            ),

            fit_metrics_aggregation_fn=(
                aggregate_fit_metrics
            ),

            evaluate_metrics_aggregation_fn=(
                aggregate_evaluate_metrics
            )

        )


    # ========================================================
    # CONFIGURE FIT
    # ========================================================

    def configure_fit(
        self,
        server_round,
        parameters,
        client_manager
    ):

        print("\n" + "=" * 70)

        print(
            f"SCAFFOLD ROUND {server_round}: "
            "CONFIGURE FIT"
        )

        print("=" * 70)


        # Flower's current global parameters
        # are MODEL parameters only.

        model_arrays = (
            parameters_to_ndarrays(
                parameters
            )
        )


        if (
            len(model_arrays)
            !=
            self.num_model_tensors
        ):

            raise ValueError(

                "Unexpected global model tensor count. "
                f"Expected "
                f"{self.num_model_tensors}, "
                f"received "
                f"{len(model_arrays)}."

            )


        # Keep exact x used at start
        # of this communication round.

        self.current_global_model = [

            array.copy()

            for array
            in model_arrays

        ]


        # Client FIT payload:
        #
        # [global model x]
        # +
        # [global control c]

        fit_payload_arrays = (

            [

                array.copy()

                for array
                in model_arrays

            ]

            +

            [

                array.copy()

                for array
                in self.global_control

            ]

        )


        fit_parameters = (
            ndarrays_to_parameters(
                fit_payload_arrays
            )
        )


        config = {

            "server_round":
                int(
                    server_round
                ),

            "server_learning_rate":
                float(
                    SERVER_LEARNING_RATE
                )

        }


        fit_ins = FitIns(

            fit_parameters,

            config

        )


        clients = client_manager.sample(

            num_clients=NUM_CLIENTS,

            min_num_clients=NUM_CLIENTS

        )


        print(
            f"Selected clients: "
            f"{len(clients)}"
        )

        print(
            f"Model tensors sent: "
            f"{self.num_model_tensors}"
        )

        print(
            f"Control tensors sent: "
            f"{self.num_control_tensors}"
        )

        print(
            "Global control L2 norm: "
            f"{l2_norm_of_arrays(self.global_control):.6f}"
        )


        return [

            (
                client,
                fit_ins
            )

            for client
            in clients

        ]


    # ========================================================
    # AGGREGATE FIT
    # ========================================================

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        print("\n" + "=" * 70)

        print(
            f"SCAFFOLD ROUND {server_round}: "
            "AGGREGATE FIT"
        )

        print("=" * 70)


        if not results:

            print(
                "No client fit results received."
            )

            return None, {}


        if failures:

            print(
                f"Client failures: "
                f"{len(failures)}"
            )


            if not self.accept_failures:

                print(
                    "Failures are not accepted."
                )

                return None, {}


        # ----------------------------------------------------
        # Collect local model y_i and Δc_i
        # ----------------------------------------------------

        local_models = []

        client_control_deltas = []

        fit_metric_input = []


        expected_payload_size = (

            self.num_model_tensors

            +

            self.num_control_tensors

        )


        for (
            client_proxy,
            fit_res
        ) in results:

            client_arrays = (
                parameters_to_ndarrays(
                    fit_res.parameters
                )
            )


            if (
                len(client_arrays)
                !=
                expected_payload_size
            ):

                raise ValueError(

                    "Invalid SCAFFOLD client payload. "
                    f"Expected "
                    f"{expected_payload_size} arrays "
                    f"({self.num_model_tensors} model + "
                    f"{self.num_control_tensors} control delta), "
                    f"received "
                    f"{len(client_arrays)}."

                )


            local_model_arrays = (

                client_arrays[
                    :
                    self.num_model_tensors
                ]

            )


            control_delta_arrays = (

                client_arrays[
                    self.num_model_tensors
                    :
                ]

            )


            local_models.append(
                local_model_arrays
            )


            client_control_deltas.append(
                control_delta_arrays
            )


            fit_metric_input.append(

                (
                    fit_res.num_examples,
                    fit_res.metrics
                )

            )


        num_participants = len(
            local_models
        )


        print(
            f"Successful clients: "
            f"{num_participants}"
        )


        # ----------------------------------------------------
        # MODEL UPDATE
        #
        # Canonical SCAFFOLD:
        #
        # x+ =
        # x + eta_g / |S|
        #       sum_i (y_i - x)
        #
        # With eta_g = 1:
        # this becomes the equal client mean.
        # ----------------------------------------------------

        new_global_model = []


        for tensor_index in range(
            self.num_model_tensors
        ):

            current_tensor = (
                self.current_global_model[
                    tensor_index
                ]
            )


            local_tensor_list = [

                client_model[
                    tensor_index
                ]

                for client_model
                in local_models

            ]


            mean_local_tensor = mean_arrays(
                local_tensor_list
            )


            # For floating-point model tensors,
            # perform the canonical server step.

            if np.issubdtype(
                current_tensor.dtype,
                np.floating
            ):

                updated_tensor = (

                    current_tensor.astype(
                        np.float64,
                        copy=False
                    )

                    +

                    SERVER_LEARNING_RATE
                    *
                    (
                        mean_local_tensor.astype(
                            np.float64,
                            copy=False
                        )

                        -

                        current_tensor.astype(
                            np.float64,
                            copy=False
                        )
                    )

                )


                updated_tensor = (
                    updated_tensor.astype(
                        current_tensor.dtype
                    )
                )


            else:

                # Integer buffers are copied
                # from the rounded equal mean.

                updated_tensor = (
                    mean_local_tensor.astype(
                        current_tensor.dtype,
                        copy=False
                    )
                )


            new_global_model.append(
                updated_tensor
            )


        # ----------------------------------------------------
        # GLOBAL CONTROL UPDATE
        #
        # SCAFFOLD:
        #
        # c+ =
        # c + (|S| / N) * mean_i(Δc_i)
        #
        # Since our experiment selects all five
        # clients each round, |S| / N = 1.
        # Keeping the general formula here makes
        # the implementation explicit.
        # ----------------------------------------------------

        participation_fraction = (

            num_participants
            /
            NUM_CLIENTS

        )


        new_global_control = []


        for control_index in range(
            self.num_control_tensors
        ):

            delta_list = [

                client_delta[
                    control_index
                ]

                for client_delta
                in client_control_deltas

            ]


            mean_delta = mean_arrays(
                delta_list
            )


            updated_control = (

                self.global_control[
                    control_index
                ].astype(
                    np.float64,
                    copy=False
                )

                +

                participation_fraction
                *
                mean_delta.astype(
                    np.float64,
                    copy=False
                )

            )


            updated_control = (
                updated_control.astype(

                    self.global_control[
                        control_index
                    ].dtype

                )
            )


            new_global_control.append(
                updated_control
            )


        self.current_global_model = [

            array.copy()

            for array
            in new_global_model

        ]


        self.latest_model = [

            array.copy()

            for array
            in new_global_model

        ]


        self.global_control = [

            array.copy()

            for array
            in new_global_control

        ]


        # ----------------------------------------------------
        # SAVE THE ACTUAL AGGREGATED MODEL
        # ----------------------------------------------------

        round_checkpoint_path = (

            CHECKPOINT_DIR
            /
            (
                "scaffold_densenet121_"
                f"round_{server_round}.pth"
            )

        )


        save_model_checkpoint(

            self.latest_model,

            round_checkpoint_path

        )


        print(
            f"Round checkpoint saved: "
            f"{round_checkpoint_path}"
        )


        print(
            "Updated global control L2 norm: "
            f"{l2_norm_of_arrays(self.global_control):.6f}"
        )


        # ----------------------------------------------------
        # AGGREGATED TRAINING METRICS
        # ----------------------------------------------------

        aggregated_metrics = (
            aggregate_fit_metrics(
                fit_metric_input
            )
        )


        if "train_loss" in aggregated_metrics:

            print(
                "Aggregated train loss: "
                f"{aggregated_metrics['train_loss']:.6f}"
            )


        # Flower must receive ONLY
        # global model parameters here.

        return (

            ndarrays_to_parameters(
                self.latest_model
            ),

            aggregated_metrics

        )


    # ========================================================
    # CONFIGURE EVALUATION
    # ========================================================

    def configure_evaluate(
        self,
        server_round,
        parameters,
        client_manager
    ):

        if self.fraction_evaluate == 0.0:

            return []


        model_arrays = (
            parameters_to_ndarrays(
                parameters
            )
        )


        if (
            len(model_arrays)
            !=
            self.num_model_tensors
        ):

            raise ValueError(

                "Evaluation received invalid "
                "model tensor count. "
                f"Expected "
                f"{self.num_model_tensors}, "
                f"received "
                f"{len(model_arrays)}."

            )


        evaluate_ins = EvaluateIns(

            parameters,

            {

                "server_round":
                    int(
                        server_round
                    )

            }

        )


        clients = client_manager.sample(

            num_clients=NUM_CLIENTS,

            min_num_clients=NUM_CLIENTS

        )


        return [

            (
                client,
                evaluate_ins
            )

            for client
            in clients

        ]


# ============================================================
# INITIALIZE MODEL AND CONTROL VARIATES
# ============================================================

initial_model = create_densenet121(
    num_classes=2
)


initial_model_arrays = (
    get_model_ndarrays(
        initial_model
    )
)


trainable_parameters = list(
    initial_model.named_parameters()
)


initial_control_arrays = [

    np.zeros(

        tuple(
            parameter.shape
        ),

        dtype=(
            parameter
            .detach()
            .cpu()
            .numpy()
            .dtype
        )

    )

    for _,
    parameter
    in trainable_parameters

]


NUM_MODEL_TENSORS = len(
    initial_model_arrays
)


NUM_CONTROL_TENSORS = len(
    initial_control_arrays
)


print("=" * 70)

print(
    "CANONICAL SCAFFOLD SERVER"
)

print("=" * 70)

print(
    f"Server address: "
    f"{SERVER_ADDRESS}"
)

print(
    f"Communication rounds: "
    f"{NUM_ROUNDS}"
)

print(
    f"Clients per round: "
    f"{NUM_CLIENTS}"
)

print(
    f"Server learning rate: "
    f"{SERVER_LEARNING_RATE}"
)

print(
    f"Model tensors: "
    f"{NUM_MODEL_TENSORS}"
)

print(
    f"Control tensors: "
    f"{NUM_CONTROL_TENSORS}"
)


# ============================================================
# CREATE STRATEGY
# ============================================================

strategy = ScaffoldStrategy(

    initial_model_arrays=(
        initial_model_arrays
    ),

    initial_control_arrays=(
        initial_control_arrays
    ),

    num_model_tensors=(
        NUM_MODEL_TENSORS
    ),

    num_control_tensors=(
        NUM_CONTROL_TENSORS
    )

)


# ============================================================
# START SERVER
# ============================================================

history = fl.server.start_server(

    server_address=(
        SERVER_ADDRESS
    ),

    config=fl.server.ServerConfig(

        num_rounds=NUM_ROUNDS

    ),

    strategy=strategy

)


# ============================================================
# SAVE TRUE FINAL AGGREGATED MODEL
# ============================================================

save_model_checkpoint(

    strategy.latest_model,

    FINAL_MODEL_PATH

)


# ============================================================
# SAVE FINAL GLOBAL CONTROL VARIATE
# ============================================================

control_state = OrderedDict()


for (
    name,
    _
), control_array in zip(

    trainable_parameters,
    strategy.global_control

):

    control_state[
        name
    ] = torch.from_numpy(

        control_array.copy()

    )


torch.save(

    control_state,

    FINAL_CONTROL_PATH

)


print("\n" + "=" * 70)

print(
    "SCAFFOLD TRAINING COMPLETED"
)

print("=" * 70)

print(
    f"Final model saved: "
    f"{FINAL_MODEL_PATH}"
)

print(
    f"Final control saved: "
    f"{FINAL_CONTROL_PATH}"
)

print(
    "Final global control L2 norm: "
    f"{l2_norm_of_arrays(strategy.global_control):.6f}"
)