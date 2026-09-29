from typing import List, Union
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .general import exponential_moving_average


def plot_cifar10(X: np.ndarray, y: np.ndarray) -> None:
    """Display a grid of random samples from the CIFAR-10 dataset.

    Args:
        X (np.ndarray): Array of images with shape (num_samples, height, width, channels).
        y (np.ndarray): Array of labels corresponding to the images.

    Returns:
        None

    Note:
        The function selects 7 random samples from each of the 10 classes in the CIFAR-10 dataset.
        Images are displayed in a grid with the class name shown above the first image of each row.
    """
    generator = torch.Generator().manual_seed(42)
    classes = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Reshape the images to 32x32 pixels
    X = X.reshape(-1, 32, 32, 3)

    num_classes = len(classes)
    samples_per_class = 7

    plt.figure(figsize=(num_classes, samples_per_class))

    for label, class_name in enumerate(classes):
        # Find indices of images for the current class
        indices = np.flatnonzero(y == label)
        indices = np.random.choice(indices, samples_per_class, replace=False)

        for i, idx in enumerate(indices):
            plt_idx = i * num_classes + label + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X[idx].astype("uint8"))
            plt.axis("off")
            if i == 0:
                plt.title(class_name)
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> None:
    """Show a confusion matrix of true labels (rows) against predicted labels (columns).

    Args:
        y_true (np.ndarray): The true labels of shape (N,).
        y_pred (np.ndarray): The predicted labels of shape (N,).
        class_names (List[str]): The name of each class, indexed by label.

    Returns:
        None

    Note:
        Each cell shows the number of samples. The color shows the share of the row,
        so classes with different numbers of samples are comparable.
    """
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    row_share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=9,
                    color="white" if row_share[i, j] > 0.5 else "black")
    ax.set_xticks(range(num_classes), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(num_classes), labels=class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    accuracy = np.trace(matrix) / max(matrix.sum(), 1)
    ax.set_title(f"Confusion matrix (accuracy {accuracy:.1%})")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of the true class")
    plt.tight_layout()
    plt.show()


def plot_knn_cross_validation(k_to_metrics: dict, label_names: list = None):
    """Show the results of cross-validation for different k values in KNN.

    Args:
        k_to_metrics (dict): A dictionary containing cross-validation results.
            - The dictionary should have keys 'accuracy', 'precision', 'recall', and 'f1'.
            - Each key should map to a dictionary where the keys are k values (as strings) and the values are lists of metric values.
        label_names (List[str], optional): A list of class labels for the precision, recall, and F1 score metrics. Default is None.

    Returns:
        None
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Cross-validation on k")

    # Plot accuracy on the first subplot
    k_acc = np.array([int(k) for k in k_to_metrics["accuracy"].keys()])
    acc_means = np.array([np.mean(v) for v in k_to_metrics["accuracy"].values()])
    acc_stds = np.array([np.std(v) for v in k_to_metrics["accuracy"].values()])
    _plot_error_bar(
        ax1,
        k_acc,
        acc_means,
        acc_stds,
        x_label="k",
        y_label="Accuracy",
        title="Accuracy",
    )

    # Plot precision on the second subplot
    k_pre = np.array([int(k) for k in k_to_metrics["precision"].keys()])
    precision_means = np.hstack(
        [v[..., np.newaxis] for v in k_to_metrics["precision"].values()]
    )
    _plot_knn_metric(ax2, k_pre, precision_means, label_names, "Precision")

    # Plot recall on the third subplot
    k_rec = np.array([int(k) for k in k_to_metrics["recall"].keys()])
    recall_means = np.hstack(
        [v[..., np.newaxis] for v in k_to_metrics["recall"].values()]
    )
    _plot_knn_metric(ax3, k_rec, recall_means, label_names, "Recall")

    # Plot f1 on the fourth subplot
    k_f1 = np.array([int(k) for k in k_to_metrics["f1"].keys()])
    f1_means = np.hstack([v[..., np.newaxis] for v in k_to_metrics["f1"].values()])
    _plot_knn_metric(ax4, k_f1, f1_means, label_names, "F1")

    plt.tight_layout()
    plt.show()


def plot_training(
    loss_history: dict, accuracy_history: dict, ema: bool = False, alpha: float = 0.1
) -> None:
    """Show the training history of a neural network.

    Args:
        loss_history (dict): Dictionary containing the training and validation loss history.
        accuracy_history (dict): Dictionary containing the training and validation accuracy history.
        ema (bool, optional): Whether to apply exponential moving average to the history. Defaults to False.
        alpha (float, optional): The alpha value for the exponential moving average. Defaults to 0.1.

    Returns:
        None

    Note:
        This function creates a plot with two subplots, displaying the loss and accuracy history.
    """
    train_loss_iters = np.array(list(loss_history["train"].keys()))
    train_losses = np.array(list(loss_history["train"].values()))

    val_loss_iters = np.array(list(loss_history["val"].keys()))
    val_losses = np.array(list(loss_history["val"].values()))

    train_acc_iters = np.array(list(accuracy_history["train"].keys()))
    train_accs = np.array(list(accuracy_history["train"].values()))

    val_acc_iters = np.array(list(accuracy_history["val"].keys()))
    val_accs = np.array(list(accuracy_history["val"].values()))

    if ema:
        train_losses = exponential_moving_average(train_losses, alpha)
        val_losses = exponential_moving_average(val_losses, alpha)
        train_accs = exponential_moving_average(train_accs, alpha)
        val_accs = exponential_moving_average(val_accs, alpha)

    X_acc = [train_acc_iters, val_acc_iters]
    Y_acc = [train_accs, val_accs]

    X_loss = [train_loss_iters, val_loss_iters]
    Y_loss = [train_losses, val_losses]

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    _plot_values(
        axes[0],
        X_loss,
        Y_loss,
        ["Training", "Validation"],
        "Iterations",
        "Loss",
        "Loss History",
    )
    _plot_values(
        axes[1],
        X_acc,
        Y_acc,
        ["Training", "Validation"],
        "Iterations",
        "Accuracy",
        "Accuracy History",
    )

    # Let matplotlib choose the iteration ticks and limits
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
        ax.autoscale(enable=True, axis="x")

    plt.show()


def plot_training_runs(
    loss_histories: dict, accuracy_histories: dict, title: str = ""
) -> None:
    """Compare several training runs: training loss and validation accuracy.

    Args:
        loss_histories (dict): Maps a run name to the loss history returned by train().
        accuracy_histories (dict): Maps a run name to the accuracy history returned by train().
        title (str, optional): Title of the whole figure. Defaults to "".

    Returns:
        None
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(loss_histories)))

    for color, name in zip(colors, loss_histories):
        train_loss = loss_histories[name]["train"]
        val_acc = accuracy_histories[name]["val"]
        ax_loss.plot(
            list(train_loss.keys()),
            [float(v) for v in train_loss.values()],
            color=color,
            linewidth=1.2,
            label=name,
        )
        ax_acc.plot(
            list(val_acc.keys()),
            list(val_acc.values()),
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=name,
        )

    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Iterations")
    ax_loss.set_ylabel("Training loss (log scale)")
    ax_loss.set_title("Training Loss")
    ax_loss.grid(True, which="both", alpha=0.4)

    ax_acc.set_xlabel("Iterations")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Validation Accuracy")
    ax_acc.grid(True, alpha=0.4)

    # One legend for both panels, to the right of the figure
    ax_acc.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def _templates_as_images(weights: torch.Tensor) -> np.ndarray:
    """Turn the weights of shape (3072, C) into C images of shape (32, 32, 3) in [0, 1].

    Every template is rescaled on its own, so its lowest weight is black and its
    highest weight is white.
    """
    w = weights.detach().cpu().numpy().T.reshape(-1, 32, 32, 3)
    w_min = w.min(axis=(1, 2, 3), keepdims=True)
    w_max = w.max(axis=(1, 2, 3), keepdims=True)
    return (w - w_min) / (w_max - w_min)


def plot_weights_as_templates(weights: torch.Tensor, class_names: List[str]) -> None:
    """Show every column of the weight matrix as a 32x32 color image.

    Args:
        weights (torch.Tensor): Weights of a linear classifier of shape (3072, C).
        class_names (List[str]): The name of each class, indexed by label.

    Returns:
        None
    """
    templates = _templates_as_images(weights)

    fig, axes = plt.subplots(2, len(class_names) // 2, figsize=(10, 4))
    for ax, template, name in zip(axes.flat, templates, class_names):
        ax.imshow(template)
        ax.set_title(name)
        ax.axis("off")

    plt.show()


def plot_template_comparison(weights_by_name: dict, class_names: List[str]) -> None:
    """Show the templates of several classifiers, one row per classifier.

    Args:
        weights_by_name (dict): Maps a row title to weights of shape (3072, C).
        class_names (List[str]): The name of each class, indexed by label.

    Returns:
        None
    """
    num_rows, num_cols = len(weights_by_name), len(class_names)
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(1.3 * num_cols, 1.45 * num_rows), squeeze=False
    )

    for row, (name, weights) in enumerate(weights_by_name.items()):
        for col, template in enumerate(_templates_as_images(weights)):
            ax = axes[row, col]
            ax.imshow(template)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(class_names[col])
            if col == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center")

    plt.tight_layout()
    plt.show()


def plot_template_matches(
    weights: torch.Tensor,
    scores: torch.Tensor,
    images: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    num_matches: int = 8,
) -> None:
    """For every class, show its template and the images with the highest score for it.

    A green frame marks an image of that class, a red frame an image of another
    class, whose true class is written below it.

    Args:
        weights (torch.Tensor): Weights of a linear classifier of shape (3072, C).
        scores (torch.Tensor): Scores of the images of shape (N, C).
        images (np.ndarray): The original images of shape (N, 32, 32, 3) with values 0-255.
        y_true (np.ndarray): The true labels of shape (N,).
        class_names (List[str]): The name of each class, indexed by label.
        num_matches (int, optional): Number of images per class. Defaults to 8.

    Returns:
        None
    """
    templates = _templates_as_images(weights)
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    num_classes = len(class_names)

    fig, axes = plt.subplots(
        num_classes, num_matches + 1, figsize=(1.25 * (num_matches + 1), 1.45 * num_classes)
    )

    for c in range(num_classes):
        axes[c, 0].imshow(templates[c])
        axes[c, 0].set_ylabel(class_names[c], rotation=0, ha="right", va="center", fontsize=12)

        best = np.argsort(-scores[:, c])[:num_matches]
        for ax, i in zip(axes[c, 1:], best):
            correct = y_true[i] == c
            ax.imshow(images[i].astype("uint8"))
            for spine in ax.spines.values():
                spine.set_edgecolor("#2CA02C" if correct else "#D62728")
                spine.set_linewidth(3)
            if not correct:
                ax.set_xlabel(class_names[y_true[i]], fontsize=9, labelpad=2, color="#D62728")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_title("Template", fontsize=11)
    axes[0, (num_matches + 1) // 2].set_title("Highest-scoring validation images", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_transform_robustness(
    example_images: dict,
    zoom_factors: List[float],
    zoom_accuracies: List[float],
    flip_accuracy: float,
) -> None:
    """Show an example image under every transformation and the accuracy on transformed images.

    Args:
        example_images (dict): Maps a title to an image of shape (32, 32, 3) with values 0-255.
        zoom_factors (List[float]): The zoom factors, 1 means no zoom.
        zoom_accuracies (List[float]): The accuracy for every zoom factor.
        flip_accuracy (float): The accuracy on horizontally flipped images.

    Returns:
        None
    """
    num_cols = int(np.ceil(len(example_images) / 2))
    fig = plt.figure(figsize=(11, 4))
    grid = fig.add_gridspec(2, num_cols + 3, width_ratios=[1] * num_cols + [0.3, 2, 2])

    for k, (title, image) in enumerate(example_images.items()):
        ax = fig.add_subplot(grid[k // num_cols, k % num_cols])
        ax.imshow(np.clip(np.asarray(image), 0, 255).astype("uint8"))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    ax = fig.add_subplot(grid[:, num_cols + 1 :])
    ax.plot(zoom_factors, zoom_accuracies, marker="o", color="#222222", label="Zoomed in")
    ax.scatter([1], [flip_accuracy], marker="*", s=200, color="#EE7733",
               zorder=3, label="Flipped left-right")
    ax.axhline(1 / 10, color="gray", linestyle=":", label="Random guessing")
    ax.set_xlabel("Zoom factor")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Accuracy on zoomed and flipped images")
    ax.set_ylim(0, None)
    ax.set_xticks(zoom_factors, labels=[f"{f:g}x" for f in zoom_factors])
    ax.grid(True, alpha=0.4)
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.show()


def _plot_knn_metric(
    ax: plt.Axes,
    k_choices: np.ndarray,
    metric_means: np.ndarray,
    label_names: List[str],
    metric_name: str,
) -> None:
    """Plot a metric across different values of k.

    Args:
        ax (plt.Axes): The subplot where the metric will be plotted.
        k_choices (np.ndarray): Array of k values.
        metric_means (np.ndarray): 2D array of metric values, with each row corresponding to a class.
        label_names (List[str]): List of labels for each class.
        metric_name (str): The name of the metric to be plotted.

    Returns:
        None
    """
    for i in range(metric_means.shape[0]):
        label = label_names[i] if label_names is not None else f"Class {i}"
        ax.plot(k_choices, metric_means[i], linewidth=1.5, marker="o", label=label)
    _set_plot(ax, k_choices, "k", metric_name, metric_name, legend=True)


def _plot_values(
    ax: plt.Axes,
    X: Union[np.ndarray, List[np.ndarray]],
    Y: Union[np.ndarray, List[np.ndarray]],
    label_names: List[str] = None,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
) -> None:
    """Plot data on a given Matplotlib Axes object.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to plot on.
        X (Union[np.ndarray, List[np.ndarray]]): The x-coordinates of the data.
        Y (Union[np.ndarray, List[np.ndarray]]): The y-coordinates of the data.
        label_names (List[str], optional): A list of labels for the legend. Default is None.
        x_label (str, optional): The label for the x-axis. Default is an empty string.
        y_label (str, optional): The label for the y-axis. Default is an empty string.
        title (str, optional): The title of the plot. Default is an empty string.

    Returns:
        None

    Note:
        If the x-coordinates have fewer than 20 elements, markers are shown on the plot.
    """
    assert len(X) == len(Y), "X and Y must have the same length"
    legend = label_names is not None and len(X) > 1

    for i in range(len(X)):
        label = label_names[i] if legend else None
        if X[i].size < 20:
            ax.plot(X[i], Y[i], linewidth=1.5, marker="o", label=label)
        else:
            ax.plot(X[i], Y[i], linewidth=1.5, label=label)
    _set_plot(ax, X, x_label, y_label, title, legend=legend)


def _plot_error_bar(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    stds: np.ndarray,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
) -> None:
    """Plot data with error bars on a given Matplotlib Axes object.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to plot on.
        x (np.ndarray): The x-coordinates of the data.
        y (np.ndarray): The y-coordinates of the data.
        stds (np.ndarray): The standard deviations for the error bars.
        x_label (str, optional): The label for the x-axis. Default is an empty string.
        y_label (str, optional): The label for the y-axis. Default is an empty string.
        title (str, optional): The title of the plot. Default is an empty string.

    Returns:
        None
    """
    ax.errorbar(
        x, y, yerr=stds, linestyle="dotted", linewidth=1.5, marker="o", capsize=4
    )
    _set_plot(ax, x, x_label, y_label, title)


def _set_plot(
    ax: plt.Axes,
    X: Union[np.ndarray, List[np.ndarray]],
    x_label: str,
    y_label: str,
    title: str,
    legend: bool = False,
) -> None:
    """Set plot settings such as labels, title, grid, and legend.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to adjust.
        X (Union[np.ndarray, List[np.ndarray]]): The x-coordinates used to determine ticks and limits.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.
        legend (bool): Whether to display a legend.

    Returns:
        None
    """
    unique_x = list(set(np.concatenate(X))) if isinstance(X, list) else X
    min_k, max_k = np.min(unique_x), np.max(unique_x)

    if legend:
        ax.legend()

    if len(unique_x) < 20:
        ax.set_xticks(unique_x)
        ax.set_xlim(min_k - 1, max_k + 0.35 * (max_k - min_k))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
