import torch.nn as nn
import torchvision.models as models


def get_resnet34(num_classes):
    """
    Instantiate a ResNet-34 model with the specified number of output classes.

    Args:
        num_classes (int): Number of classes for classification.

    Returns:
        torchvision.models.ResNet: Instantiated ResNet-34 model with modified fully connected layer.
    """
    model = models.resnet34(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
