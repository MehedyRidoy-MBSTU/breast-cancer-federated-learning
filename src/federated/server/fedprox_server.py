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

MODEL_NAME = "DenseNet-121"


CHECKPOINT_DIR = Path(
    "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


GLOBAL_MODEL_PATH = (
    CHECKPOINT_DIR /
    "fedprox_densenet121_final.pth"
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
# MODEL FUNCTIONS
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
# INITIAL MODEL
# ============================================================


global_model = create_model()


initial_parameters = (

    fl.common
    .ndarrays_to_parameters(

        get_parameters(
            global_model
        )

    )

)



# ============================================================
# FEDPROX STRATEGY
# ============================================================


class FedProxStrategy(
    fl.server.strategy.FedProx
):


    def __init__(self):

        super().__init__(

            fraction_fit=1.0,

            fraction_evaluate=1.0,


            min_fit_clients=NUM_CLIENTS,

            min_evaluate_clients=NUM_CLIENTS,

            min_available_clients=NUM_CLIENTS,


            proximal_mu=0.01,


            initial_parameters=initial_parameters

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


            print("\n")
            print("="*70)

            print(
                f"FEDPROX ROUND {server_round} COMPLETED"
            )

            print("="*70)



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


            checkpoint = (

                CHECKPOINT_DIR /

                f"fedprox_densenet121_round_{server_round}.pth"

            )


            torch.save(

                model.state_dict(),

                checkpoint

            )


            print(
                "Saved checkpoint:"
            )

            print(
                checkpoint
            )


        return aggregated_parameters




    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures
    ):


        result = (

            super()
            .aggregate_evaluate(

                server_round,

                results,

                failures

            )

        )


        if result[0] is not None:

            print(
                f"FedProx evaluation loss round {server_round}:"
            )

            print(
                result[0]
            )


        return result



# ============================================================
# SERVER START
# ============================================================


def main():


    print("="*70)

    print(
        "FEDPROX SERVER - DENSENET-121"
    )

    print("="*70)


    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Rounds: {NUM_ROUNDS}"
    )

    print(
        f"Clients: {NUM_CLIENTS}"
    )


    strategy = FedProxStrategy()



    history = fl.server.start_server(

        server_address="127.0.0.1:8080",


        config=fl.server.ServerConfig(

            num_rounds=NUM_ROUNDS

        ),


        strategy=strategy

    )



    print("\n")
    print("="*70)

    print(
        "FEDPROX TRAINING COMPLETED"
    )

    print("="*70)



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



if __name__ == "__main__":

    main()