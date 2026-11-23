import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18():
    # Create ResNet-18 with random weights
    model = resnet18(weights=None)

    # Our SAR images have 2 channels (VV and VH), not 3 (RGB)
    model.conv1 = nn.Conv2d(
        2, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # We have 3 classes: Oil, No_oil, Lookalike
    model.fc = nn.Linear(model.fc.in_features, 3)

    return model