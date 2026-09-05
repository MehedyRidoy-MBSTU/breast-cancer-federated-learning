from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BreaKHisDataset(Dataset):
    """
    PyTorch Dataset for the BreaKHis histopathological image dataset.

    Expected CSV columns:
        path
        filename
        biopsy_procedure
        tumor_class
        tumor_type
        year
        slide_id
        magnification
        sequence
        specimen_id
        label

    Binary classification:
        B = 0 (Benign)
        M = 1 (Malignant)
    """

    def __init__(
        self,
        csv_file,
        transform=None,
        image_size=224,
    ):
        """
        Parameters
        ----------
        csv_file : str or Path
            Path to train.csv, validation.csv, or test.csv.

        transform : torchvision.transforms.Compose, optional
            Image transformations.

        image_size : int
            Target image size if no custom transform is provided.
        """

        self.csv_file = Path(csv_file)

        # Load CSV
        self.data = pd.read_csv(self.csv_file)

        # --------------------------------------------------------
        # Validate required columns
        # --------------------------------------------------------

        required_columns = [
            "path",
            "filename",
            "tumor_class",
            "tumor_type",
            "magnification",
            "specimen_id",
            "label",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in {self.csv_file}: "
                f"{missing_columns}"
            )

        # --------------------------------------------------------
        # Default transformation
        # --------------------------------------------------------

        if transform is None:

            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

        else:
            self.transform = transform

    # ============================================================
    # NUMBER OF SAMPLES
    # ============================================================

    def __len__(self):
        return len(self.data)

    # ============================================================
    # LOAD ONE IMAGE
    # ============================================================

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(row["path"])

        # Check image exists
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Binary label
        label = int(row["label"])

        return image, torch.tensor(
            label,
            dtype=torch.long
        )

    # ============================================================
    # GET IMAGE INFORMATION
    # ============================================================

    def get_metadata(self, index):

        return self.data.iloc[index].to_dict()