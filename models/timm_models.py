import torch
import timm
import torch.nn as nn
import torchvision.models as models


def timm_efficient_net(num_classes):
    """
    Instantiate a ResNet-34 model with the specified number of output classes.

    Args:
        num_classes (int): Number of classes for classification.

    Returns:
        torchvision.models.ResNet: Instantiated ResNet-34 model with modified fully connected layer.
    """
    model = timm.create_model("efficientnet_b0", pretrained=True)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)


    return model
