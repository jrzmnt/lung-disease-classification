import torch.nn as nn
import torchvision.models as models


def get_mobile_net(num_classes):
    """
    Instantiate a MobileNetV2 model with the specified number of output classes.

    Args:
        num_classes (int): Number of classes for classification.

    Returns:
        torchvision.models.MobileNetV2: Instantiated MobileNetV2 model with modified classifier.
    """
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model
