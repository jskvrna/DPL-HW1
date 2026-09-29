from typing import Union, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier


class Data2DVisualizer:
    """
    A class for visualizing 2D datasets and their splits, including decision boundaries.

    This class allows you to visualize datasets and their splits, such as training, validation, and test sets.
    It also provides the capability to display decision boundaries for classifiers.

    Args:
        train_split (Tuple[np.ndarray, np.ndarray]):
            A tuple containing the training data features and labels.

        val_split (Tuple[np.ndarray, np.ndarray], optional):
            A tuple containing the validation data features and labels. Defaults to (np.array([]), np.array([])).

        test_split (Tuple[np.ndarray, np.ndarray], optional):
            A tuple containing the test data features and labels. Defaults to (np.array([]), np.array([])).

    Properties:
        x_lim (Tuple[float, float]): Returns the x-axis limits for the plot.
        y_lim (Tuple[float, float]): Returns the y-axis limits for the plot.
        num_classes (int): Returns the number of unique classes in the dataset.
        num_splits (int): Returns the number of dataset splits available.

    Methods:
        show_dataset: Displays a visualization of dataset splits using subplots for training, validation, and test sets.
        show_decision_boundaries: Displays decision boundaries for a classifier.
        show_knn_principle: Illustrates the k-Nearest Neighbors (k-NN) principle with a plot.
    """

    def __init__(
        self,
        train_split: Tuple[
            Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]
        ],
        val_split: Tuple[
            Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]
        ] = (np.array([]), np.array([])),
        test_split: Tuple[
            Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]
        ] = (np.array([]), np.array([])),
    ):
        self.X_train, self.y_train = train_split
        self.X_val, self.y_val = val_split
        self.X_test, self.y_test = test_split

        self.val = True if self.X_test.shape[0] != 0 else False
        self.test = True if self.X_train.shape[0] != 0 else False

        self.features = np.concatenate([self.X_train, self.X_val, self.X_test])
        self.labels = np.concatenate([self.y_train, self.y_val, self.y_test])

        self.colors = [
            [0, 0.4, 1],
            [1, 0, 0.4],
            [0, 1, 0.5],
            [1, 0.7, 0.5],
            "violet",
            "mediumaquamarine",
        ]
        self.color_map = ListedColormap(self.colors[: self.num_classes])

    @property
    def x_lim(self) -> Tuple[float, float]:
        """Get the x-axis limits for the plot.

        Returns:
            Tuple[float, float]: A tuple containing the minimum and maximum values of the x-axis.
        """

        min_x, max_x = np.min(self.features[:, 0]), np.max(self.features[:, 0])
        range_x = max_x - min_x
        return min_x - range_x * 0.1, max_x + range_x * 0.1

    @property
    def y_lim(self) -> Tuple[float, float]:
        """Get the y-axis limits for the plot.

        Returns:
            Tuple[float, float]: A tuple containing the minimum and maximum values of the y-axis.
        """

        min_y, max_y = np.min(self.features[:, 1]), np.max(self.features[:, 1])
        range_y = max_y - min_y
        return min_y - range_y * 0.1, max_y + range_y * 0.1

    @property
    def num_classes(self) -> int:
        """Get the number of unique classes in the dataset.

        Returns:
            int: The number of unique classes.
        """

        return np.size(np.unique(self.labels))

    @property
    def num_splits(self) -> int:
        """Get the number of dataset splits available.

        Returns:
            int: The number of dataset splits (1 + validation + test).
        """

        return 1 + self.val + self.test

    @property
    def lut(self) -> np.ndarray:
        """
        Example for 3 classes
        cmap = np.zeros((255, 4))

        # first third is first color
        cmap[:85, :] = np.array(self.color_map(0)) * 255

        # second third is second color
        cmap[85:170, :] = np.array(self.color_map(1)) * 255

        # last third is third color
        cmap[170:, :] = np.array(self.color_map(2)) * 255
        """

        lut = np.zeros((255, 4))

        step = 255 // self.num_classes

        for i in range(self.num_classes):
            lut[i * step : (i + 1) * step, :] = np.array(self.color_map(i)) * 255

        return lut

    def show_dataset(self) -> None:
        """Displays a visualization of dataset splits using subplots for training,
        validation, and test sets.

        Returns:
            None
        """

        fig, axes = plt.subplots(1, self.num_splits, figsize=(self.num_splits * 6, 6))
        fig.suptitle("Dataset Splits", fontsize=30)

        self._plot_data(self.X_train, self.y_train, axes[0], title="Training")
        self._plot_data(self.X_val, self.y_val, axes[1], title="Validation")
        self._plot_data(self.X_test, self.y_test, axes[2], title="Test")

        plt.tight_layout()
        plt.show()

    def show_decision_boundaries(
        self, classifier, h: float = 0.001, transform=None, show_test: bool = True
    ) -> None:
        """Display decision boundaries for a classifier.

        This function plots decision boundaries for a given classifier along with the dataset splits.

        Args:
            classifier: The trained classifier for which decision boundaries will be plotted.
            h (float, optional): Step size for meshgrid. Smaller values create finer boundaries. Default is 0.001.
            transform (callable, optional): Maps the 2D points to the features the classifier expects.
            show_test (bool, optional): Also show the test split. Defaults to True.

        Returns:
            None
        """

        splits = [
            ("Training", self.X_train, self.y_train),
            ("Validation", self.X_val, self.y_val),
        ]
        if show_test:
            splits.append(("Test", self.X_test, self.y_test))

        fig, axes = plt.subplots(1, len(splits), figsize=(len(splits) * 6, 6))
        fig.suptitle("Decision Boundaries", fontsize=30)

        for ax, (title, features, labels) in zip(axes, splits):
            self._plot_data(features, labels, ax, title=title)
            self._plot_decision_boundaries(classifier, h, ax, transform=transform)

        plt.tight_layout()
        plt.show()

    def compare_decision_boundaries(
        self, classifiers: dict, h: float = 0.01, confidence: bool = False
    ) -> None:
        """Show the decision boundaries of several classifiers side by side.

        Every panel shows the training data and the regions of one classifier.

        Args:
            classifiers (dict): Maps a panel title to a trained classifier.
            h (float, optional): Step size for meshgrid. Defaults to 0.01.
            confidence (bool, optional): Fade each region where the classifier is
                unsure (low softmax probability). Only for PyTorch models. Defaults to False.

        Returns:
            None
        """

        n = len(classifiers)
        fig, axes = plt.subplots(1, n, figsize=(n * 4.5, 4.5), squeeze=False)

        for ax, (title, classifier) in zip(axes[0], classifiers.items()):
            self._plot_data(self.X_train, self.y_train, ax, title=title)
            ax.title.set_fontsize(14)
            self._plot_decision_boundaries(classifier, h, ax, confidence=confidence)

        plt.tight_layout()
        plt.show()

    def show_knn_principle(self) -> None:
        """Illustrate the k-Nearest Neighbors (k-NN) principle with a plot.

        This function creates a plot demonstrating the k-NN principle using a sample data point and its neighbors.

        Returns:
            None
        """

        fig, ax = plt.subplots(figsize=(6, 6))
        self._plot_data(self.X_train, self.y_train, ax)

        p = np.array([0.5, 0.5])

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.X_train, self.y_train)

        neighbor_indices = knn.kneighbors([p], return_distance=False)[0]

        for i in neighbor_indices:
            ax.plot(
                [p[0], self.X_train[i, 0]],
                [p[1], self.X_train[i, 1]],
                color="gray",
                linestyle="--",
                linewidth=2,
            )
        ax.scatter(
            p[0],
            p[1],
            color="#ffd500",
            s=100,
            linewidth=1.5,
            zorder=10,
            edgecolor="black",
        )

        fig.savefig("knn_principle.png", dpi=70, bbox_inches="tight", pad_inches=0.1)

    def show_linear_weights(self, classifier) -> None:
        """Draw the weight vector of every class on top of the decision regions.

        Only differences between the class scores matter, so every arrow shows
        w_c minus the mean of all weight vectors. The arrows start at the mean of
        the training data and keep their relative lengths.

        Args:
            classifier: A trained linear classifier with params["W"] of shape (2, C).

        Returns:
            None
        """

        fig, ax = plt.subplots(figsize=(6, 6))
        self._plot_data(self.X_train, self.y_train, ax, title="Class weight vectors")
        self._plot_decision_boundaries(classifier, 0.01, ax)

        weights = classifier.params["W"].detach().numpy().T
        directions = weights - weights.mean(axis=0)

        # Scale the arrows so the longest one covers 35 % of the plot width
        scale = 0.35 * (self.x_lim[1] - self.x_lim[0]) / np.max(np.linalg.norm(directions, axis=1))
        origin = np.asarray(self.X_train, dtype=float).mean(axis=0)

        for c, direction in enumerate(directions):
            ax.annotate(
                "",
                xy=origin + scale * direction,
                xytext=origin,
                arrowprops=dict(
                    arrowstyle="-|>,head_width=0.5,head_length=1",
                    linewidth=3,
                    facecolor=self.colors[c],
                    edgecolor="black",
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=20,
            )
            ax.text(
                *(origin + 1.12 * scale * direction),
                f"$w_{c}$",
                fontsize=16,
                ha="center",
                va="center",
                zorder=21,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        ax.scatter(*origin, color="white", edgecolor="black", s=60, zorder=22)
        plt.show()

    def show_decision_functions(self, classifier) -> None:
        """Illustrate the decision functions with a plot.
        This function creates a plot demonstrating the decision functions of a classifier.

        Args:
        classifier: The trained classifier for which decision functions will be plotted.

        Returns:
        None
        """
        # A 60 x 60 grid rounded to 3 decimals keeps the saved notebook small
        x_rng = np.linspace(self.x_lim[0], self.x_lim[1], 60).round(3)
        y_rng = np.linspace(self.y_lim[0], self.y_lim[1], 60).round(3)
        xx, yy = np.meshgrid(x_rng, y_rng)
        X = np.column_stack((xx.ravel(), yy.ravel()))
        X = torch.from_numpy(X).float()

        with torch.no_grad():
            logits = classifier.forward(X)
        logits = logits.numpy()

        fig = go.Figure()

        # Plot decision surfaces
        for c in range(self.num_classes):
            zz = logits[:, c].reshape(xx.shape).round(3)
            color = f"rgb{tuple(int(val * 255) for val in self.colors[c])}"

            fig.add_trace(
                go.Surface(
                    x=x_rng,
                    y=y_rng,
                    z=zz,
                    colorscale=[[0, color], [1, color]],
                    opacity=1,
                    showscale=False,
                    name=f"Class {c} Decision Function",
                    showlegend=True,
                )
            )

        # Plot training points
        y_train = (
            self.y_train
            if isinstance(self.y_train, np.ndarray)
            else self.y_train.numpy()
        )
        for c in range(self.num_classes):
            mask = y_train == c
            color = f"rgb{tuple(int(val * 255) for val in self.colors[c])}"

            fig.add_trace(
                go.Scatter3d(
                    x=self.X_train[mask, 0],
                    y=self.X_train[mask, 1],
                    z=np.zeros(np.sum(mask)),
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=color,
                        line=dict(width=1, color="black"),  # width of the border
                    ),
                    name=f"Class {c} Samples",
                )
            )

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Decision Function",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        fig.show()

    def show_hidden_features(self, classifier) -> None:
        """Show where the training points end up after every layer of an MLP.

        The first panel is the input with the decision regions of the whole
        network. Every other panel shows the output of one hidden layer: directly
        if the layer has 2 neurons, otherwise projected to 2D with PCA. The last
        panel also shows the regions of the output layer, which is a linear
        classifier working on the features of the last hidden layer.

        The points are colored by class; pale points lie near the center of the
        spiral, strong points near its outer end.

        Args:
            classifier: A trained MLPClassifier.

        Returns:
            None
        """

        X = torch.as_tensor(self.X_train, dtype=torch.float)
        hidden = self._layer_outputs(classifier, X)[:-1]
        colors = self._radius_shaded_colors()

        n = len(hidden) + 1
        fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))

        ax = axes[0]
        self._plot_decision_boundaries(classifier, 0.01, ax)
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=25, linewidths=0.4, edgecolors="black", zorder=10)
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_title(f"Input ({X.shape[1]} features)", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        for layer, (ax, features) in enumerate(zip(axes[1:], hidden), start=1):
            self._plot_hidden_layer(classifier, features, layer, layer == len(hidden), colors, ax)

        plt.tight_layout()
        plt.show()

    def compare_hidden_features(self, classifiers: dict) -> None:
        """Show the last hidden layer of several MLPs side by side.

        Args:
            classifiers (dict): Maps a panel title to a trained MLPClassifier.

        Returns:
            None
        """

        X = torch.as_tensor(self.X_train, dtype=torch.float)
        colors = self._radius_shaded_colors()

        n = len(classifiers)
        fig, axes = plt.subplots(1, n, figsize=(n * 5, 5), squeeze=False)

        for ax, (title, classifier) in zip(axes[0], classifiers.items()):
            hidden = self._layer_outputs(classifier, X)[:-1]
            self._plot_hidden_layer(classifier, hidden[-1], len(hidden), True, colors, ax)
            ax.set_title(f"{title}: hidden layer {len(hidden)}", fontsize=14)

        plt.tight_layout()
        plt.show()

    def show_neuron_activations(self, classifier, num_neurons: int = 4) -> None:
        """Show what single neurons of an MLP compute, as images over the input plane.

        Every row is one layer. Bright means a large value of the neuron. From the
        first hidden layer only the neurons with the largest outgoing weights are
        shown. The last row shows the softmax probability of every class.

        Args:
            classifier: A trained MLPClassifier.
            num_neurons (int, optional): Maximum number of neurons per layer. Defaults to 4.

        Returns:
            None
        """

        x = np.linspace(self.x_lim[0], self.x_lim[1], 200)
        y = np.linspace(self.y_lim[0], self.y_lim[1], 200)
        xx, yy = np.meshgrid(x, y)
        grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

        outputs = self._layer_outputs(classifier, grid)
        outputs[-1] = torch.softmax(outputs[-1], dim=1)
        num_layers = len(outputs)
        num_cols = max(num_neurons, self.num_classes)

        fig, axes = plt.subplots(num_layers, num_cols, figsize=(num_cols * 3, num_layers * 3), squeeze=False)

        for layer, values in enumerate(outputs, start=1):
            values = values.numpy()
            is_output = layer == num_layers

            if is_output:
                neurons = list(range(values.shape[1]))
            elif values.shape[1] <= num_neurons:
                neurons = list(range(values.shape[1]))
            else:
                # The neurons the next layer listens to the most
                W_next = classifier.params[f"W{layer + 1}"].detach()
                importance = torch.linalg.norm(W_next, dim=1)
                neurons = sorted(torch.argsort(importance, descending=True)[:num_neurons].tolist())

            for col, ax in enumerate(axes[layer - 1]):
                ax.set_xticks([])
                ax.set_yticks([])
                if col >= len(neurons):
                    ax.axis("off")
                    continue

                neuron = neurons[col]
                image = values[:, neuron].reshape(xx.shape)
                ax.imshow(
                    image,
                    origin="lower",
                    extent=(x[0], x[-1], y[0], y[-1]),
                    aspect="auto",
                    cmap="viridis",
                    vmin=0 if is_output else None,
                    vmax=1 if is_output else None,
                )
                ax.scatter(
                    self.X_train[:, 0],
                    self.X_train[:, 1],
                    c=self.y_train,
                    cmap=self.color_map,
                    s=4,
                    alpha=0.5,
                    linewidths=0,
                )
                if is_output:
                    ax.set_title(f"Output: p(class {neuron})", fontsize=12)
                else:
                    ax.set_title(f"Layer {layer}, neuron {neuron + 1}", fontsize=12)

        plt.tight_layout()
        plt.show()

    def _plot_hidden_layer(
        self, classifier, features: torch.Tensor, layer: int, is_last: bool, colors: np.ndarray, ax: plt.Axes
    ) -> None:
        """Scatter the features of one hidden layer, with the output regions if it is the last one."""

        size = features.shape[1]
        if size == 2:
            points = features.numpy()
            ax.set_title(f"Hidden layer {layer} ({size} neurons)", fontsize=14)
            ax.set_xlabel("neuron 1")
            ax.set_ylabel("neuron 2")
        else:
            # PCA: project onto the two directions with the largest spread
            centered = features - features.mean(dim=0)
            _, _, directions = torch.linalg.svd(centered, full_matrices=False)
            points = (centered @ directions[:2].T).numpy()
            ax.set_title(f"Hidden layer {layer} ({size} neurons, PCA to 2D)", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

        low, high = points.min(axis=0), points.max(axis=0)
        pad = 0.08 * (high - low) + 1e-6
        ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
        ax.set_ylim(low[1] - pad[1], high[1] + pad[1])

        if is_last and size == 2:
            self._plot_output_layer_regions(classifier, ax)

        ax.scatter(points[:, 0], points[:, 1], c=colors, s=25, linewidths=0.4, edgecolors="black", zorder=10)

    def _layer_outputs(self, classifier, X: torch.Tensor) -> list:
        """Run an MLP layer by layer and return the output of every layer.

        Uses the parameters W1, b1, W2, b2, ... of the classifier and applies
        the activation after every layer except the last one.

        Returns:
            list: The hidden features of every hidden layer, followed by the logits.
        """

        num_layers = sum(1 for name in classifier.params if name.startswith("W"))
        outputs = []
        with torch.no_grad():
            h = X
            for layer in range(1, num_layers + 1):
                h = h @ classifier.params[f"W{layer}"] + classifier.params[f"b{layer}"]
                if layer < num_layers:
                    h = classifier.activation_func(h)
                outputs.append(h)
        return outputs

    def _plot_output_layer_regions(self, classifier, ax: plt.Axes) -> None:
        """Color the current axis by the prediction of the output layer alone."""

        num_layers = sum(1 for name in classifier.params if name.startswith("W"))
        W = classifier.params[f"W{num_layers}"].detach()
        b = classifier.params[f"b{num_layers}"].detach()

        x = np.linspace(*ax.get_xlim(), 300)
        y = np.linspace(*ax.get_ylim(), 300)
        xx, yy = np.meshgrid(x, y)
        grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
        predictions = torch.argmax(grid @ W + b, dim=1).numpy().reshape(xx.shape)

        ax.pcolormesh(xx, yy, predictions, alpha=0.4, cmap=self.color_map, vmin=0, vmax=self.num_classes - 1)

    def _radius_shaded_colors(self) -> np.ndarray:
        """Class colors of the training points, pale near the center, strong far from it."""

        X = np.asarray(self.X_train, dtype=float)
        radius = np.linalg.norm(X - X.mean(axis=0), axis=1)
        strength = 0.15 + 0.85 * (radius - radius.min()) / (radius.max() - radius.min())

        base = self.color_map(np.asarray(self.y_train, dtype=int))[:, :3]
        return strength[:, None] * base + (1 - strength[:, None]) * np.ones(3)

    def _plot_data(
        self,
        features: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        ax: plt.Axes,
        title: str = "",
    ):
        """Plot data points with labels.

        This internal function is used to plot data points with labels and adjust the plot settings.

        Parameters:
            features (numpy.ndarray): The features of the data points.
            labels (numpy.ndarray): The labels of the data points.
            ax (matplotlib.pyplot.Axes): The axis for plotting.
            title (str, optional): The title of the plot. Default is an empty string.

        Returns:
            None
        """

        ax.set_xlim(self.x_lim[0], self.x_lim[1])
        ax.set_ylim(self.y_lim[0], self.y_lim[1])
        ax.axis("off")

        scatter_args = dict(
            s=100,
            alpha=0.85,
            c=labels,
            linewidths=1.5,
            edgecolors="black",
            cmap=self.color_map,
            zorder=10,
        )

        ax.scatter(features[:, 0], features[:, 1], **scatter_args)
        ax.set_title(title, fontsize=20)

    def _plot_decision_boundaries(
        self,
        classifier,
        h: float,
        ax: plt.Axes,
        transform=None,
        confidence: bool = False,
    ) -> None:
        """Plot decision boundaries.

        This internal function is used to plot decision boundaries for a given classifier.

        Parameters:
            classifier: The trained classifier for which decision boundaries will be plotted.
            h (float): Step size for meshgrid. Smaller values create finer boundaries.
            ax (matplotlib.pyplot.Axes): The axis for plotting.
            transform (callable, optional): Maps the 2D points to the features the classifier expects.
            confidence (bool, optional): Fade the regions where the softmax probability is low.

        Returns:
            None
        """

        x = np.arange(self.x_lim[0], self.x_lim[1], h)
        y = np.arange(self.y_lim[0], self.y_lim[1], h)
        xx, yy = np.meshgrid(x, y)

        mesh_matrix = np.c_[xx.ravel(), yy.ravel()]

        # PyTorch models (with a forward method) take tensors, k-NN takes numpy arrays
        if hasattr(classifier, "forward"):
            mesh_matrix = torch.from_numpy(mesh_matrix).float()
            if transform is not None:
                mesh_matrix = transform(mesh_matrix)
            with torch.no_grad():
                mesh_predictions = classifier.predict(mesh_matrix).numpy()
                if confidence:
                    probs = torch.softmax(classifier.forward(mesh_matrix), dim=1)
                    certainty = probs.max(dim=1).values.numpy()
        else:
            if transform is not None:
                mesh_matrix = transform(mesh_matrix)
            mesh_predictions = classifier.predict(mesh_matrix)

        mesh_predictions = mesh_predictions.reshape(xx.shape)

        if not confidence:
            ax.pcolormesh(xx, yy, mesh_predictions, alpha=0.4, cmap=self.color_map)
            return

        # Opacity grows from 0.1 (all classes equally likely) to 0.6 (certain),
        # so the regions stay visible even for a very unsure classifier
        chance = 1 / self.num_classes
        opacity = 0.1 + 0.5 * np.clip((certainty - chance) / (1 - chance), 0, 1)
        image = self.color_map(mesh_predictions.ravel().astype(int))
        image[:, 3] = opacity.ravel()
        ax.imshow(
            image.reshape(*xx.shape, 4),
            origin="lower",
            extent=(x[0], x[-1], y[0], y[-1]),
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
