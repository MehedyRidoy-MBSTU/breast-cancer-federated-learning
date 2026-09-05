import sys
import flwr as fl

from src.federated.clients.densenet_client import DenseNetClient


SERVER_ADDRESS = "127.0.0.1:8080"


if __name__ == "__main__":


    if len(sys.argv) != 2:

        print(
            "Usage: python -m src.federated.clients.run_client client_id"
        )

        sys.exit(1)


    client_id = sys.argv[1]


    print("=" * 70)
    print("FEDERATED DENSENET-121 CLIENT")
    print("=" * 70)


    print(f"Client ID: {client_id}")
    print(f"Server: {SERVER_ADDRESS}")


    client = DenseNetClient(
        client_id
    )


    print("\nClient initialized successfully")

    print(
        f"Training samples: {len(client.train_dataset)}"
    )

    print(
        f"Evaluation samples: {len(client.eval_dataset)}"
    )


    print("\nConnecting to Flower server...")


    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
    )


    print("\nClient finished")