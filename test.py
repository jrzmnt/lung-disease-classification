"""
Script for testing a trained model.

This script loads a pre-trained model, sets up data loaders for testing, and evaluates the model's performance
using the provided test dataset.

Usage:
    python test_model.py --model <model_name>

Options:
    --model <model_name>       Name of the pre-trained model to test.

Example:
    python test_model.py --model resnet34
"""

import torch
import pickle
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from models.simple_cnn import get_simple_cnn
from models.mobile_net import get_mobile_net
from models.shuffle_net import get_shuffle_net
from models.resnet18 import get_resnet18
from models.resnet34 import get_resnet34
from models.timm_models import timm_efficient_net
from datasets.image_dataset import ImageDataset, prepare_datasets
from utils.transformations import get_transforms
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# GLOBAL VARIABLES (HYPERPARAMS)
BATCH_SIZE = 128


def load_model(model_name, num_classes):
    """
    Load a pre-trained model.

    Args:
        model_name (str): Name of the model to load.
        num_classes (int): Number of output classes for the model.

    Returns:
        torch.nn.Module: The loaded pre-trained model.

    Raises:
        ValueError: If an invalid model name is provided.

    Example:
        model = load_model("resnet34", num_classes=3)
    """
    if model_name == "simple-cnn":
        model = get_simple_cnn(num_classes)
        model.load_state_dict(torch.load("weights/simple-cnn.pth"))
    elif model_name == "mobile-net":
        model = get_mobile_net(num_classes)
        model.load_state_dict(torch.load("weights/mobile-net.pth"))
    elif model_name == "shuffle-net":
        model = get_shuffle_net(num_classes)
        model.load_state_dict(torch.load("weights/shuffle-net.pth"))
    elif model_name == "resnet18":
        model = get_resnet18(num_classes)
        model.load_state_dict(torch.load("weights/resnet18.pth"))
    elif model_name == "resnet34":
        model = get_resnet34(num_classes)
        model.load_state_dict(torch.load("weights/resnet34.pth"))
    elif model_name == "efficient-net":
        model = timm_efficient_net(num_classes)
        model.load_state_dict(torch.load("weights/efficient-net.pth"))
    else:
        raise ValueError("Invalid model name")

    return model


def setup_data_loaders():
    """
    Set up data loaders for training, validation, and testing.

    Args:
        data_paths (list): List of paths to data directories.
        class_labels (list): List of class labels.

    Returns:
        tuple: A tuple containing three data loaders: (train_loader, valid_loader, test_loader).

    Example:
        data_paths = ["data/Normal", "data/Lung_Opacity", "data/Viral Pneumonia/"]
        class_labels = ["Normal", "Lung Opacity", "Viral Pneumonia"]
        train_loader, valid_loader, test_loader = setup_data_loaders(data_paths, class_labels)
    """
    data_paths = ["data/Normal", "data/Lung_Opacity", "data/Viral Pneumonia/"]
    class_labels = ["Normal", "Lung Opacity", "Viral Pneumonia"]

    train_df, val_df, test_df = prepare_datasets(data_paths, class_labels)

    train_dataset = ImageDataset(train_df, transform=get_transforms())
    valid_dataset = ImageDataset(val_df, transform=get_transforms())
    test_dataset = ImageDataset(test_df, transform=get_transforms())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader


def test(model, test_loader, model_name):
    """
    Test the specified model using the provided test data loader.

    Args:
        model: The model to be tested.
        test_loader: DataLoader for test data.
        model_name (str): Name of the model.

    Returns:
        dict: A dictionary containing the test results including accuracy, precision, recall,
        F1-score, confusion matrix, image paths, predicted labels, and true labels.

    Example:
        results = test(model, test_loader, "resnet34")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc=f"Testing {model_name}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 50)
    print(f"Test Results for Model: {model_name}")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print("=" * 50 + "\n")

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "image_paths": all_paths,
        "predictions": all_preds,
        "labels": all_labels,
    }

    with open(f"logs/test_results_{model_name}.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained model")
    parser.add_argument("--model", type=str, required=True, help="Model to test")
    args = parser.parse_args()

    train_loader, valid_loader, test_loader = setup_data_loaders()

    model = load_model(args.model, num_classes=3)
    test(model, test_loader, args.model)
