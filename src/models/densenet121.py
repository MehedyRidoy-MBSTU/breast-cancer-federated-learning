import torch
import torch.nn as nn
from torchvision.models import (
    densenet121,
    DenseNet121_Weights
)


def create_densenet121(num_classes=2):

    model = densenet121(
        weights=DenseNet121_Weights.DEFAULT
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        num_classes
    )

    return model


if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = create_densenet121()

    model = model.to(device)

    x = torch.randn(
        2, 3, 224, 224
    ).to(device)

    with torch.no_grad():
        y = model(x)

    print("DenseNet-121 created successfully")
    print(f"Device: {device}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {parameters:,}"
    )