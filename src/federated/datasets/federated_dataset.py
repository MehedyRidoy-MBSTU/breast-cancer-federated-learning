"""
Federated BreaKHis Dataset Loader

Purpose:
    Load individual client datasets for Federated Learning.

Each client receives:
    - Separate CSV file
    - Separate specimen distribution
    - Local DataLoader

No data sharing between clients.

Compatible with:
    Flower + PyTorch
"""


from pathlib import Path

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from torchvision import transforms



# ============================================================
# FEDERATED DATASET
# ============================================================


class FederatedBreaKHisDataset(Dataset):

    """
    Dataset loader for a single federated client.

    Input:
        client_1.csv
        client_2.csv
        ...

    Expected columns:

        path
        filename
        label
        specimen_id
        magnification

    """


    def __init__(
        self,
        csv_file,
        transform=None,
        image_size=224
    ):

        self.csv_file = Path(csv_file)


        if not self.csv_file.exists():

            raise FileNotFoundError(
                f"Dataset CSV not found: {self.csv_file}"
            )


        self.data = pd.read_csv(
            self.csv_file
        )


        required_columns = [

            "path",
            "filename",
            "label"

        ]


        missing = [

            c for c in required_columns

            if c not in self.data.columns

        ]


        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )


        if transform is None:


            self.transform = transforms.Compose(

                [

                    transforms.Resize(
                        (image_size, image_size)
                    ),

                    transforms.ToTensor()

                ]

            )

        else:

            self.transform = transform



    # ========================================================
    # LENGTH
    # ========================================================


    def __len__(self):

        return len(self.data)



    # ========================================================
    # GET ITEM
    # ========================================================


    def __getitem__(self, index):


        row = self.data.iloc[index]


        image_path = Path(
            row["path"]
        )


        if not image_path.exists():

            raise FileNotFoundError(
                f"Image missing: {image_path}"
            )


        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        label = int(
            row["label"]
        )


        if self.transform:

            image = self.transform(
                image
            )


        return image, label



# ============================================================
# CLIENT DATALOADER CREATOR
# ============================================================


def create_client_dataloader(

    client_id,

    batch_size=32,

    shuffle=True,

    num_workers=0,

    transform=None

):


    """

    Create DataLoader for a federated client.

    Example:

        client_1.csv

    """


    csv_path = (

        Path("processed")

        /
        "federated"

        /
        f"{client_id}.csv"

    )


    dataset = FederatedBreaKHisDataset(

        csv_path,

        transform=transform

    )


    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers

    )


    return loader



# ============================================================
# TEST
# ============================================================


if __name__ == "__main__":



    print("=" * 70)

    print(
        "FEDERATED DATASET LOADER TEST"
    )

    print("=" * 70)



    clients = [

        "client_1",

        "client_2",

        "client_3",

        "client_4",

        "client_5"

    ]



    for client in clients:


        loader = create_client_dataloader(

            client_id=client,

            batch_size=32

        )


        images, labels = next(
            iter(loader)
        )


        print()

        print(client)

        print(
            "Samples:",
            len(loader.dataset)
        )

        print(
            "Batch images:",
            images.shape
        )

        print(
            "Batch labels:",
            labels.shape
        )



    print()

    print("=" * 70)

    print(
        "FEDERATED DATASET TEST COMPLETED"
    )

    print("=" * 70)