"""
This script provides utility functions and classes for handling image datasets, preparing datasets for training, and loading pre-trained models for image classification tasks.

Imports:
    - os: Operating system-specific functionality.
    - pandas as pd: Data manipulation and analysis library.
    - torch: PyTorch library for deep learning.
    - train_test_split from sklearn.model_selection: Split arrays or matrices into random train and test subsets.
    - Dataset, DataLoader from torch.utils.data: Dataset and DataLoader classes for managing data loading in PyTorch.
    - Image from PIL: Image processing library.
    - ImageDataset: Custom dataset class for loading image data from a DataFrame.
    - prepare_datasets: Function to prepare training, validation, and test datasets from provided data paths and class labels.

Classes:
    - ImageDataset: Custom dataset class for loading image data from a DataFrame.

Functions:
    - prepare_datasets: Function to prepare training, validation, and test datasets from provided data paths and class labels.

"""

import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Custom dataset class for loading image data from a DataFrame.

        Args:
            dataframe (DataFrame): DataFrame containing file paths and labels.
            transform (callable, optional): Optional transform to be applied to the images.

        Attributes:
            dataframe (DataFrame): DataFrame containing file paths and labels.
            transform (callable, optional): Optional transform to be applied to the images.
            label_map (dict): Mapping of class labels to numeric indices.

        """
        self.dataframe = dataframe
        self.transform = transform
        self.label_map = {"Normal": 0, "Lung Opacity": 1, "Viral Pneumonia": 2}

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves an item (image, label, file path) from the dataset at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image, label, and file path.
        """
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path)
        label_str = self.dataframe.iloc[idx, 1]
        label = self.label_map[label_str]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label), img_path


def prepare_datasets(data_paths, class_labels):
    """
    Prepare training, validation, and test datasets from provided data paths and class labels.

    Args:
        data_paths (list): List of paths to data directories.
        class_labels (list): List of class labels.

    Returns:
        tuple: A tuple containing DataFrames for training, validation, and test datasets.

    """
    file_paths_list = []
    labels_list = []

    for i, data_path in enumerate(data_paths):
        files = os.listdir(data_path)
        for filename in files:
            file_path = os.path.join(data_path, filename)
            file_paths_list.append(file_path)
            labels_list.append(class_labels[i])

    file_paths_series = pd.Series(file_paths_list, name="filepaths")
    labels_series = pd.Series(labels_list, name="labels")
    data = pd.concat([file_paths_series, labels_series], axis=1)
    df = pd.DataFrame(data)

    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df.labels
    )

    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=42, stratify=train_df.labels
    )

    print(f"train: {train_df.shape}, test: {test_df.shape}, validation: {val_df.shape}")

    return train_df, val_df, test_df
