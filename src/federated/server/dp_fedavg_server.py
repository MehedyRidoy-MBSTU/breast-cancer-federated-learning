import warnings
warnings.filterwarnings("ignore")


from pathlib import Path
from collections import OrderedDict

import torch
import flwr as fl


from src.models.densenet121 import create_densenet121



# ============================================================
# CONFIGURATION
# ============================================================

NUM_ROUNDS = 5

NUM_CLIENTS = 5


MODEL_NAME = "DP-FedAvg DenseNet-121"


CHECKPOINT_DIR = Path(
    "checkpoints"
)


RESULTS_DIR = Path(
    "results/dp_fedavg"
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)



GLOBAL_MODEL_PATH = (

    CHECKPOINT_DIR

    /

    "dp_fedavg_densenet121_final.pth"

)



# ============================================================
# DIFFERENTIAL PRIVACY CONFIGURATION
# ============================================================

DP_MAX_GRAD_NORM = 1.0

DP_NOISE_MULTIPLIER = 1.0



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
# MODEL UTILITIES
# ============================================================

def create_model():

    return create_densenet121(
        num_classes=2
    )



def get_parameters(model):

    return [

        value.detach()
        .cpu()
        .numpy()

        for _, value in model.state_dict().items()

    ]



def set_parameters(
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
# METRIC AGGREGATION
# ============================================================

def fit_metrics_aggregation_fn(
    metrics
):

    total_examples = sum(

        num_examples

        for num_examples, _ in metrics

    )


    train_loss = sum(

        num_examples *
        metric["train_loss"]

        for num_examples, metric in metrics

    ) / total_examples



    return {

        "train_loss":

        train_loss

    }



def evaluate_metrics_aggregation_fn(
    metrics
):

    total_examples = sum(

        num_examples

        for num_examples, _ in metrics

    )


    accuracy = sum(

        num_examples *
        metric["accuracy"]

        for num_examples, metric in metrics

    ) / total_examples



    f1 = sum(

        num_examples *
        metric["f1"]

        for num_examples, metric in metrics

    ) / total_examples



    roc_auc = sum(

        num_examples *
        metric["roc_auc"]

        for num_examples, metric in metrics

    ) / total_examples



    return {

        "accuracy":

        accuracy,


        "f1":

        f1,


        "roc_auc":

        roc_auc

    }



# ============================================================
# INITIAL PARAMETERS
# ============================================================

global_model = create_model()


initial_parameters = (

    fl.common.ndarrays_to_parameters(

        get_parameters(
            global_model
        )

    )

)



# ============================================================
# DP-FEDAVG STRATEGY
# ============================================================

class DPFedAvgStrategy(
    fl.server.strategy.FedAvg
):


    def __init__(self):

        super().__init__(

            fraction_fit=1.0,


            fraction_evaluate=1.0,


            min_fit_clients=3,


            min_evaluate_clients=3,


            min_available_clients=3,


            initial_parameters=
            initial_parameters,


            fit_metrics_aggregation_fn=
            fit_metrics_aggregation_fn,


            evaluate_metrics_aggregation_fn=
            evaluate_metrics_aggregation_fn

        )



    def aggregate_fit(

        self,

        server_round,

        results,

        failures

    ):


        aggregated_parameters = (

            super()

            .aggregate_fit(

                server_round,

                results,

                failures

            )

        )


        if aggregated_parameters[0] is not None:


            print()

            print(
                "=" * 70
            )


            print(
                f"DP-FEDAVG ROUND {server_round} COMPLETED"
            )


            print(
                "=" * 70
            )


            parameters = (

                fl.common

                .parameters_to_ndarrays(

                    aggregated_parameters[0]

                )

            )


            model = create_model()


            set_parameters(

                model,

                parameters

            )



            round_path = (

                CHECKPOINT_DIR

                /

                f"dp_fedavg_densenet121_round_{server_round}.pth"

            )


            torch.save(

                model.state_dict(),

                round_path

            )


            print(
                "Saved checkpoint:"
            )


            print(
                round_path
            )



        return aggregated_parameters



# ============================================================
# SERVER START
# ============================================================

def main():


    print(
        "=" * 70
    )


    print(
        "DP-FEDAVG SERVER - DENSENET-121"
    )


    print(
        "=" * 70
    )


    print(
        f"Device: {DEVICE}"
    )


    print(
        f"Rounds: {NUM_ROUNDS}"
    )


    print(
        f"Clients: {NUM_CLIENTS}"
    )


    print(
        "\nDifferential Privacy:"
    )


    print(
        f"Gradient clipping norm: {DP_MAX_GRAD_NORM}"
    )


    print(
        f"Noise multiplier: {DP_NOISE_MULTIPLIER}"
    )



    strategy = DPFedAvgStrategy()



    fl.server.start_server(

        server_address=
        "127.0.0.1:8080",


        config=

        fl.server.ServerConfig(

            num_rounds=
            NUM_ROUNDS

        ),


        strategy=
        strategy

    )



    print()

    print(
        "=" * 70
    )


    print(
        "DP-FEDAVG TRAINING COMPLETED"
    )


    print(
        "=" * 70
    )



    final_model = create_model()



    torch.save(

        final_model.state_dict(),

        GLOBAL_MODEL_PATH

    )


    print(
        "Final model saved:"
    )


    print(
        GLOBAL_MODEL_PATH
    )



# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()