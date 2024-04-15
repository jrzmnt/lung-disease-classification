"""
This module provides a function to define data augmentation and normalization transformations using torchvision.

Functions:
    get_transforms(): Defines data augmentation and normalization transformations.

Example:
    transform = get_transforms()
"""

from torchvision import transforms


def get_transforms():
    """
    Defines data augmentation and normalization transformations.

    Returns:
        torchvision.transforms.Compose: A composition of data augmentation and normalization transforms.

    Example:
        transform = get_transforms()
    """
    mean, std = [0.2860, 0.5660, 0.4400], [0.1837, 0.2486, 0.1257]

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(),
        ]
    )

    return transform


def get_simple_transforms():
    """
    Defines a simple transformation.

    Returns:
        torchvision.transforms.Compose: A composition of data augmentation and normalization transforms.

    Example:
        transform = get_simple_transforms()
    """
    mean, std = [0.2860, 0.5660, 0.4400], [0.1837, 0.2486, 0.1257]

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transform
