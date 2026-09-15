import sys

import flwr as fl

from src.federated.clients.scaffold_client import ScaffoldClient


# ============================================================
# CONFIGURATION
# ============================================================

SERVER_ADDRESS = "127.0.0.1:8080"

VALID_CLIENTS = {

    "client_1",
    "client_2",
    "client_3",
    "client_4",
    "client_5",

}


# ============================================================
# COMMAND LINE ARGUMENT
# ============================================================

if len(sys.argv) != 2:

    print(
        "Usage:\n"
        "python -m "
        "src.federated.clients.run_scaffold_client "
        "client_1"
    )

    sys.exit(1)


CLIENT_ID = sys.argv[1]


if CLIENT_ID not in VALID_CLIENTS:

    raise ValueError(

        "Invalid client ID.\n"
        "Use one of:\n"
        "client_1\n"
        "client_2\n"
        "client_3\n"
        "client_4\n"
        "client_5"

    )


# ============================================================
# CREATE CLIENT
# ============================================================

print("=" * 70)

print(
    f"STARTING CANONICAL SCAFFOLD CLIENT: "
    f"{CLIENT_ID}"
)

print("=" * 70)

print(
    f"Server: {SERVER_ADDRESS}"
)


client = ScaffoldClient(
    client_id=CLIENT_ID
)


# ============================================================
# CONNECT TO SERVER
# ============================================================

fl.client.start_client(

    server_address=SERVER_ADDRESS,

    client=client.to_client()

)