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


CHECKPOINT_DIR = Path(
    "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



GLOBAL_MODEL_PATH = (
    CHECKPOINT_DIR /
    "scaffold_densenet121_final.pth"
)



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
# MODEL
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
# SCAFFOLD STRATEGY
# ============================================================

class ScaffoldStrategy(
    fl.server.strategy.FedAvg
):


    def aggregate_fit(

        self,

        server_round,

        results,

        failures

    ):


        aggregated = super().aggregate_fit(

            server_round,

            results,

            failures

        )


        if aggregated[0] is not None:


            print("\n")
            print("=" * 70)

            print(
                f"SCAFFOLD ROUND {server_round} COMPLETED"
            )

            print("=" * 70)



            parameters = (

                fl.common
                .parameters_to_ndarrays(

                    aggregated[0]

                )

            )



            model = create_model()



            state_dict = OrderedDict(

                {

                    key:

                    torch.tensor(value)

                    for key, value in zip(

                        model.state_dict().keys(),

                        parameters

                    )

                }

            )


            model.load_state_dict(
                state_dict
            )


            checkpoint = (

                CHECKPOINT_DIR /

                f"scaffold_densenet121_round_{server_round}.pth"

            )


            torch.save(

                model.state_dict(),

                checkpoint

            )


            print(
                "Saved checkpoint:"
            )

            print(checkpoint)



        return aggregated



# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":


    print("=" * 70)

    print(
        "SCAFFOLD SERVER - DENSENET-121"
    )

    print("=" * 70)


    print(
        f"Device: {DEVICE}"
    )


    print(
        f"Rounds: {NUM_ROUNDS}"
    )


    print(
        f"Clients: {NUM_CLIENTS}"
    )



    strategy = ScaffoldStrategy(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=NUM_CLIENTS,

        min_evaluate_clients=NUM_CLIENTS,

        min_available_clients=NUM_CLIENTS,

        initial_parameters=initial_parameters

    )



    fl.server.start_server(

        server_address="127.0.0.1:8080",

        config=fl.server.ServerConfig(

            num_rounds=NUM_ROUNDS

        ),

        strategy=strategy

    )



    torch.save(

        global_model.state_dict(),

        GLOBAL_MODEL_PATH

    )


    print("\n")

    print("=" * 70)

    print(
        "SCAFFOLD TRAINING COMPLETED"
    )

    print("=" * 70)

    print(
        "Final model saved:"
    )

    print(
        GLOBAL_MODEL_PATH
    )