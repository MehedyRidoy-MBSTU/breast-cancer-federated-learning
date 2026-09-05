import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


def create_efficientnet_b0(num_classes=2):

    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
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


    model = create_efficientnet_b0()

    model = model.to(device)


    x = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)


    with torch.no_grad():
        y = model(x)


    print("EfficientNet-B0 created successfully")
    print(f"Device: {device}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")


    params = sum(
        p.numel()
        for p in model.parameters()
    )


    print(
        f"Parameters: {params:,}"
    )