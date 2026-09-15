import sys

import flwr as fl

from src.federated.clients.dp_densenet_client import (
    DP_DenseNetClient
)


# ============================================================
# CONFIGURATION
# ============================================================

SERVER_ADDRESS = "127.0.0.1:8080"



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    if len(sys.argv) != 2:

        print(
            "Usage: python -m src.federated.clients.run_dp_fedavg_client client_id"
        )

        sys.exit(1)



    client_id = sys.argv[1]



    print("=" * 70)
    print("DP-FEDAVG DENSENET-121 CLIENT")
    print("=" * 70)



    print(
        f"Client ID: {client_id}"
    )


    print(
        f"Server: {SERVER_ADDRESS}"
    )



    client = DP_DenseNetClient(
        client_id
    )



    print(
        "\nDP Client initialized successfully"
    )


    print(
        f"Training samples: {len(client.train_dataset)}"
    )


    print(
        f"Evaluation samples: {len(client.eval_dataset)}"
    )


    print(
        "\nDifferential Privacy Configuration:"
    )


    print(
        "Gradient clipping norm: 1.0"
    )


    print(
        "Noise multiplier: 1.0"
    )



    print(
        "\nConnecting to Flower server..."
    )



    fl.client.start_client(

        server_address=SERVER_ADDRESS,

        client=client.to_client(),

    )



    print(
        "\nDP-FedAvg client finished"
    )