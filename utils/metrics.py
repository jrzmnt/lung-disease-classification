"""
This module provides functions to visualize various metrics and data related to model training and evaluation.

Functions:
- plot_metrics(accuracy_df, precision_df, recall_df, f1_score_df): Plots the comparison of different metrics (accuracy, precision, recall, and F1-score) in bar charts.
- plot_confusion_matrix(cm, title="Confusion Matrix"): Plots the confusion matrix.
- plot_selected_images(image_paths, predicted_labels, true_labels, label_mapping, selected_indices): Plots selected images with their predicted and true labels.
- plot_train_loss(training_metrics): Plots the training loss curves for different models.
- plot_valid_loss(training_metrics): Plots the validation loss for each model over epochs.
- plot_valid_accuracies(training_metrics): Plots the validation accuracies for each model over epochs.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def plot_metrics(accuracy_df, precision_df, recall_df, f1_score_df):
    """
    Plots the comparison of different metrics (accuracy, precision, recall, and F1-score) in bar charts.

    Args:
        accuracy_df (DataFrame): DataFrame containing accuracy scores for different models.
        precision_df (DataFrame): DataFrame containing precision scores for different models.
        recall_df (DataFrame): DataFrame containing recall scores for different models.
        f1_score_df (DataFrame): DataFrame containing F1-score scores for different models.

    Example:
        plot_metrics(accuracy_df, precision_df, recall_df, f1_score_df)
    """

    # Defining colors for each metric
    colors = ["skyblue", "lightgreen", "#FFB380", "lightcoral"]

    # Setting up the size and layout of subplots
    plt.figure(figsize=(8, 6))

    # adding values on each bar
    def add_values(ax):
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(
                f"{height:.3f}", (x + width / 2, y + height + 0.02), ha="center"
            )

    # Plot for Accuracy
    plt.subplot(2, 2, 1)  # 2 rows, 2 columns, position 1
    ax = accuracy_df.T.plot(kind="bar", legend=False, ax=plt.gca(), color=colors[0])
    plt.title("Accuracy Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)  # Limiting the y-axis
    plt.xticks(rotation=45)
    add_values(ax)

    # Plot for Precision
    plt.subplot(2, 2, 2)  # 2 rows, 2 columns, position 2
    ax = precision_df.T.plot(kind="bar", legend=False, ax=plt.gca(), color=colors[1])
    plt.title("Precision Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    add_values(ax)

    # Plot for Recall
    plt.subplot(2, 2, 3)  # 2 rows, 2 columns, position 3
    ax = recall_df.T.plot(kind="bar", legend=False, ax=plt.gca(), color=colors[2])
    plt.title("Recall Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    add_values(ax)

    # Plot for F1-Score
    plt.subplot(2, 2, 4)  # 2 rows, 2 columns, position 4
    ax = f1_score_df.T.plot(kind="bar", legend=False, ax=plt.gca(), color=colors[3])
    plt.title("F1-Score Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    add_values(ax)

    # Adjust the layout
    plt.tight_layout()

    # Show the plots
    plt.show()


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Plots the confusion matrix.

    Args:
        cm (array): Confusion matrix.
        title (str, optional): Title for the plot. Defaults to "Confusion Matrix".

    Example:
        cm = np.array([[10, 2, 3],
                       [4, 15, 6],
                       [7, 8, 20]])
        plot_confusion_matrix(cm, title="My Confusion Matrix")
    """
    plt.figure(figsize=(8, 6))
    class_labels = ["Normal", "Lung Opacity", "Viral Pneumonia"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="flare",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title(f"{title}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.title(f"{title} for ResNet34", fontsize=16, y=1.05)
    plt.xlabel("Predicted Label", fontsize=14, labelpad=15)
    plt.ylabel("True Label", fontsize=14, labelpad=15)

    plt.show()


def plot_selected_images(
    image_paths, predicted_labels, true_labels, label_mapping, selected_indices
):
    """
    Plots selected images with their predicted and true labels.

    Args:
        image_paths (list): List of paths to images.
        predicted_labels (list): Predicted labels for the images.
        true_labels (list): True labels for the images.
        label_mapping (dict): Dictionary mapping label indices to label names.
        selected_indices (list): List of indices of images to be plotted.

    Example:
        image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
        predicted_labels = [0, 1]
        true_labels = [1, 1]
        label_mapping = {0: "Normal", 1: "Lung Opacity", 2: "Viral Pneumonia"}
        selected_indices = [0, 1]
        plot_selected_images(image_paths, predicted_labels, true_labels, label_mapping, selected_indices)
    """
    plt.figure(figsize=(15, 10))

    for i, idx in enumerate(selected_indices, 1):
        img = Image.open("../" + image_paths[idx])
        pred_label = label_mapping[predicted_labels[idx]]
        true_label = label_mapping[true_labels[idx]]

        plt.subplot(1, len(selected_indices), i)
        plt.imshow(img, cmap="gray")
        plt.title(f"Predicted: {pred_label}\nTrue: {true_label}")
        plt.axis("off")

    plt.show()


def plot_train_loss(training_metrics):
    """
    Plots the training loss curves for different models.

    Args:
    training_metrics (dict): A dictionary containing training metrics for each model.
        The keys are model names, and the values are dictionaries containing training metrics.
        Each training metrics dictionary should contain "train_losses", a list of training losses
        for each epoch.

    Returns:
    None
    """
    plt.figure(figsize=(10, 5))
    for model, metrics in training_metrics.items():
        epochs = range(1, len(metrics["train_losses"]) + 1)
        plt.plot(epochs, metrics["train_losses"], label=model)
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_valid_loss(training_metrics):
    """
    Plots the validation loss for each model over epochs.

    Args:
    training_metrics (dict): A dictionary containing training metrics for each model.
        Each key represents a model, and the corresponding value is another dictionary
        containing metrics including validation losses.

    Example:
        training_metrics = {
            'model1': {'valid_losses': [0.5, 0.4, 0.3, 0.2]},
            'model2': {'valid_losses': [0.6, 0.5, 0.4, 0.3]}
        }
        plot_valid_loss(training_metrics)
    """
    plt.figure(figsize=(10, 5))
    for model, metrics in training_metrics.items():
        epochs = range(1, len(metrics["valid_losses"]) + 1)
        plt.plot(epochs, metrics["valid_losses"], label=model)
    plt.title("Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_valid_accuracies(training_metrics):
    """
    Plots the validation accuracies for each model over epochs.

    Args:
    training_metrics (dict): A dictionary containing training metrics for each model.
        Each key represents a model, and the corresponding value is another dictionary
        containing metrics including validation accuracies.

    Example:
        training_metrics = {
            'model1': {'valid_accuracies': [80, 85, 90, 95]},
            'model2': {'valid_accuracies': [75, 80, 85, 90]}
        }
        plot_valid_accuracies(training_metrics)
    """
    plt.figure(figsize=(10, 5))
    for model, metrics in training_metrics.items():
        epochs = range(1, len(metrics["valid_accuracies"]) + 1)
        plt.plot(epochs, metrics["valid_accuracies"], label=model)
    plt.title("Validation Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
