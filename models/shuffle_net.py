import torch.nn as nn
import torchvision.models as models


def get_shuffle_net(num_classes):
    """
    Instantiate a ShuffleNet model with the specified number of output classes.

    Args:
        num_classes (int): Number of classes for classification.

    Returns:
        torchvision.models.ShuffleNet: Instantiated ShuffleNet model with modified fully connected layer.
    """
    model = models.shufflenet_v2_x1_0(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
