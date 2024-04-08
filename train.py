"""
This script trains a deep learning model using the specified architecture and dataset.

It provides functions to load the data, define and train the model, and save the training logs.

Usage:
    python train.py --model [model_name]

Arguments:
    --model (str): Name of the model to be trained. Choices: simple-cnn, mobile-net, shuffle-net.

Example:
    python train.py --model resnet34
"""

import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from models.simple_cnn import get_simple_cnn
from models.mobile_net import get_mobile_net
from models.shuffle_net import get_shuffle_net
from models.resnet18 import get_resnet18
from models.resnet34 import get_resnet34
from utils.transformations import get_transforms
from torch.utils.data import DataLoader
from datasets.image_dataset import ImageDataset, prepare_datasets

# GLOBAL VARIABLES (HYPERPARAMS)
BATCH_SIZE = 128
NUM_EPOCHS = 100
LR_PATIENCE = 5
PATIENCE = 10
LR = 0.001


def get_model(model_name, num_classes):
    """
    Retrieves a pre-defined neural network model based on the provided model name.

    Args:
        model_name (str): Name of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        torch.nn.Module: A PyTorch neural network model.

    Raises:
        ValueError: If an invalid model name is provided.

    Example:
        model = get_model("resnet18", num_classes=3)
    """
    if model_name == "simple-cnn":
        return get_simple_cnn(num_classes)
    elif model_name == "mobile-net":
        return get_mobile_net(num_classes)
    elif model_name == "shuffle-net":
        return get_shuffle_net(num_classes)
    elif model_name == "resnet18":
        return get_resnet18(num_classes)
    elif model_name == "resnet34":
        return get_resnet34(num_classes)
    else:
        raise ValueError("Invalid model name")


def setup_data_loaders():
    """
    Set up data loaders for training, validation, and testing.

    Returns:
        tuple: A tuple containing three data loaders: (train_loader, valid_loader, test_loader).

    Example:
        train_loader, valid_loader, test_loader = setup_data_loaders()
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

def save_logs(train_losses, valid_losses, valid_accuracies, lr_rates, logs_path="logs/logs"):
    """
    Save training logs to a file.

    Args:
        train_losses (list): List of training losses.
        valid_losses (list): List of validation losses.
        valid_accuracies (list): List of validation accuracies.
        lr_rates (list): List of learning rates.
        logs_path (str): Path to save the logs file. Default is "logs/logs".

    Example:
        save_logs(train_losses, valid_losses, valid_accuracies, lr_rates)
    """
    results = {
    "train_losses": train_losses,
    "valid_losses": valid_losses,
    "valid_accuracies": valid_accuracies,
    "learning_rates": lr_rates}

    with open(f"{logs_path}.pkl", "wb") as f:
        pickle.dump(results, f)

def train(model, train_loader, valid_loader, model_name):
    """
    Trains the specified model using the given data loaders.

    Args:
        model: The model to be trained.
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        model_name (str): Name of the model.

    Example:
        train(model, train_loader, valid_loader, "resnet34")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=LR_PATIENCE, factor=0.3
    )

    num_epochs = NUM_EPOCHS
    best_loss = float("inf")

    patience = PATIENCE
    epochs_no_improve = 0

    # metrics
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    lr_rates = []

    print(f'\n{100*'*'}\n{38*' '}STARTING TRAINING PHASE!{38*' '}\n{100*'*'}\n')

    for epoch in range(num_epochs):
        # train loop
        model.train()
        train_loss = 0.0

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation loop
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(
                valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # calculating the accuracy
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        accuracy = 100 * correct / total

        # collecting the metrics
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(accuracy)
        lr_rates.append(optimizer.param_groups[0]['lr'])

        # Updating scheduler
        scheduler.step(valid_loss)

        print(
            f"-> Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Valid Loss: {valid_loss:.5f}, Valid Accuracy: {accuracy:.2f}%\n"
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"weights/{model_name}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break

    torch.cuda.empty_cache()

    # saving metrics on a pickle
    todays_date = datetime.now().strftime('%d_%m_%Y')
    logs_path = f'logs/{model_name}_{todays_date}'
    save_logs(train_losses, valid_losses, valid_accuracies, lr_rates, logs_path)
    print(f"\nTraining complete. Results saved in {logs_path}.pkl!\n\n\n")


def main(model_name):
    """
    Main function to train a deep learning model.

    Args:
        model_name (str): Name of the model to be trained.

    Example:
        main("resnet34")
    """
    num_classes = 3
    model = get_model(model_name, num_classes)
    train_loader, valid_loader, test_loader = setup_data_loaders()

    train(model, train_loader, valid_loader, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to train: simple-cnn, mobile-net, shuffle-net",
    )
    args = parser.parse_args()

    main(args.model)
