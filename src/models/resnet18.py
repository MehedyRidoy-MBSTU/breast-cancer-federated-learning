import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_resnet18(num_classes=2, pretrained=True):

    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(weights=weights)

    # Original ImageNet classifier:
    # 512 → 1000
    #
    # Our task:
    # 512 → 2
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    # Create model
    model = create_resnet18(
        num_classes=2,
        pretrained=True
    )

    model = model.to(device)

    print("\nModel created successfully.")

    # Test input
    x = torch.randn(
        2,
        3,
        224,
        224,
        device=device
    )

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print("\nInput shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nOutput:")
    print(output)

    print("\nModel device:")
    print(next(model.parameters()).device)