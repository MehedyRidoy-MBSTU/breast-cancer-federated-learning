import sys
import flwr as fl

from src.federated.clients.densenet_client import (
    create_flower_client
)


SERVER_ADDRESS = "127.0.0.1:8080"


def main():

    if len(sys.argv) != 2:
        print(
            "Usage: python -m src.federated.run_client client_1"
        )
        return


    client_id = sys.argv[1]

    print("=" * 70)
    print("STARTING FLOWER CLIENT")
    print("=" * 70)

    print(f"Client ID: {client_id}")
    print(f"Server: {SERVER_ADDRESS}")


    client = create_flower_client(
        client_id
    )


    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client,
    )


if __name__ == "__main__":
    main()