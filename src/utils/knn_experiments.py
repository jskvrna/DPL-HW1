"""Plots and batched evaluations for the experiments in knn_part_1.ipynb."""

from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from assignments.knn_classifier import KNNClassifier


CLASS_COLORS = ["#4477AA", "#EE6677", "#228833"]
SIZE_COLORS = ["#4477AA", "#228833", "#CC6677", "#AA4499"]


def _predict_in_batches(model, features, batch_size=256):
    # Some BLAS builds emit spurious overflow warnings from matmul; results are unaffected.
    with np.errstate(all="ignore"):
        return np.concatenate([
            model.predict(features[start:start + batch_size])
            for start in range(0, len(features), batch_size)
        ])


def _grid(arrays, resolution=121, equal_ranges=False):
    points = np.concatenate(arrays)
    lower, upper = points.min(axis=0), points.max(axis=0)
    if equal_ranges:
        center = (lower + upper) / 2
        radius = np.max(upper - lower) / 2
        lower, upper = center - radius, center + radius
    padding = 0.08 * (upper - lower)
    xx, yy = np.meshgrid(
        np.linspace(lower[0] - padding[0], upper[0] + padding[0], resolution),
        np.linspace(lower[1] - padding[1], upper[1] + padding[1], resolution),
    )
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def _boundary(ax, xx, yy, prediction, features, labels, errors=None):
    cmap = ListedColormap(CLASS_COLORS)
    ax.contourf(xx, yy, prediction.reshape(xx.shape), levels=[-0.5, 0.5, 1.5, 2.5],
                cmap=cmap, alpha=0.20)
    ax.scatter(features[:, 0], features[:, 1], c=labels, cmap=cmap,
               vmin=0, vmax=2, s=9, alpha=0.65, linewidths=0)
    if errors is not None:
        ax.scatter(features[errors, 0], features[errors, 1], s=34,
                   facecolors="none", edgecolors="black", linewidths=0.8)
    ax.set(xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")


def run_noise_experiment(X_train, y_train, X_val, y_val,
                         noise_levels=(0, 0.5, 1.5, 3), k=3, seed=0):
    """Show training data, held-out predictions, and boundaries at each noise level."""
    levels = np.asarray(noise_levels)
    rng = np.random.default_rng(seed)
    feature_std = X_train.std(axis=0)
    train_noise = rng.uniform(-1, 1, X_train.shape) * feature_std
    val_noise = rng.uniform(-1, 1, X_val.shape) * feature_std
    datasets = [(X_train + a * train_noise, X_val + a * val_noise) for a in levels]
    xx, yy, grid = _grid([points for pair in datasets for points in pair])
    accuracy = np.zeros((len(levels), 2))
    fig, axes = plt.subplots(2, len(levels), figsize=(4 * len(levels), 8),
                             squeeze=False, layout="constrained")
    fig.suptitle(f"Coordinate noise · k = {k}\nTop: training data. Bottom: validation data; circles mark mistakes.")
    for column, (strength, (train, val)) in enumerate(zip(levels, datasets)):
        model = KNNClassifier(k=k, vectorized=True)
        model.train(train, y_train)
        train_prediction = _predict_in_batches(model, train)
        val_prediction = _predict_in_batches(model, val)
        region = _predict_in_batches(model, grid)
        accuracy[column] = [np.mean(train_prediction == y_train), np.mean(val_prediction == y_val)]
        _boundary(axes[0, column], xx, yy, region, train, y_train)
        _boundary(axes[1, column], xx, yy, region, val, y_val, val_prediction != y_val)
        axes[0, column].set_title(f"Noise {strength:g}\nTraining accuracy: {accuracy[column, 0]:.1%}")
        axes[1, column].set_title(f"Validation accuracy: {accuracy[column, 1]:.1%}")
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(levels, accuracy[:, 0], "o-", color="#222222", label="Training (each point is its own neighbor)")
    ax.plot(levels, accuracy[:, 1], "o-", color="#EE7733", label="Validation")
    ax.set(title="Accuracy as coordinate noise increases", xlabel="Noise strength a",
           ylabel="Accuracy", ylim=(0, 1.03), xticks=levels)
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()
    return {"noise_levels": levels, "accuracy": accuracy}


def run_scaling_experiment(X_train, y_train, X_val, y_val,
                           scale_factors=(1, 10, 100, 10000), k=3):
    """Show the actual raw and normalized geometry with equal coordinate units."""
    factors = np.asarray(scale_factors)
    accuracy = np.zeros((2, len(factors), 2))
    fig, axes = plt.subplots(2, len(factors), figsize=(4 * len(factors), 8),
                             squeeze=False, layout="constrained")
    fig.suptitle(f"Scaling x · k = {k}\nActual feature coordinates, with equal units on both axes. Circles mark validation mistakes.")
    for column, factor in enumerate(factors):
        scaled_train = X_train * [factor, 1]
        scaled_val = X_val * [factor, 1]
        mean, std = scaled_train.mean(axis=0), scaled_train.std(axis=0)
        std = np.where(std == 0, 1, std)
        for row, normalize in enumerate((False, True)):
            if normalize:
                train, val = [(x - mean) / std for x in (scaled_train, scaled_val)]
            else:
                train, val = scaled_train, scaled_val
            xx, yy, query_grid = _grid([train, val], equal_ranges=True)
            model = KNNClassifier(k=k, vectorized=True)
            model.train(train, y_train)
            train_prediction = _predict_in_batches(model, train)
            val_prediction = _predict_in_batches(model, val)
            region = _predict_in_batches(model, query_grid)
            accuracy[row, column] = [np.mean(train_prediction == y_train), np.mean(val_prediction == y_val)]
            _boundary(axes[row, column], xx, yy, region, train, y_train)
            axes[row, column].scatter(val[:, 0], val[:, 1], c=y_val,
                                     cmap=ListedColormap(CLASS_COLORS), vmin=0, vmax=2,
                                     marker="^", s=18, edgecolors="white", linewidths=0.4)
            wrong = val_prediction != y_val
            axes[row, column].scatter(val[wrong, 0], val[wrong, 1], s=42,
                                     facecolors="none", edgecolors="black", linewidths=0.8)
            name = "Normalized" if normalize else "Raw"
            axes[row, column].set(xlabel="Normalized x" if normalize else "Scaled x",
                                  ylabel="Normalized y" if normalize else "y")
            axes[row, column].set_title(
                f"x × {factor:,} · {name}\nTrain: {accuracy[row, column, 0]:.1%} · Val: {accuracy[row, column, 1]:.1%}"
            )
    fig.legend(handles=[
        Line2D([], [], marker="o", color="0.4", linestyle="none", label="Training point"),
        Line2D([], [], marker="^", color="0.4", linestyle="none", label="Validation point"),
    ], loc="outside lower center", ncol=2)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for row, (name, style) in enumerate((("Raw", "-"), ("Normalized", "--"))):
        for split, (label, color) in enumerate((("Training", "#222222"), ("Validation", "#EE7733"))):
            ax.plot(factors, accuracy[row, :, split], marker="o", linestyle=style,
                    color=color, label=f"{label} · {name.lower()} features")
    ax.set(title="Changing units changes raw k-NN distances", xlabel="Multiplier of x",
           ylabel="Accuracy", ylim=(0, 1.03), xscale="log")
    ax.set_xticks(factors, labels=[f"{n:,}" for n in factors])
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()
    return {"scale_factors": factors, "accuracy": accuracy}


def _merge_neighbors(best_distances, best_labels, distances, labels, k):
    """Retain the exact k closest points seen so far, without sorting a full row."""
    local_k = min(k, distances.shape[1])
    indices = np.argpartition(distances, local_k - 1, axis=1)[:, :local_k].copy()
    candidates = np.concatenate([best_distances, np.take_along_axis(distances, indices, axis=1)], axis=1)
    candidate_labels = np.concatenate([best_labels, labels[indices]], axis=1)
    order = np.argsort(candidates, axis=1, kind="stable")[:, :k]
    return (np.take_along_axis(candidates, order, axis=1),
            np.take_along_axis(candidate_labels, order, axis=1))


def _vote(distances, labels, k):
    """Apply the student's voting method to each query's retained neighbors."""
    voter = KNNClassifier(k=k, vectorized=True)
    predictions = np.empty(len(distances), dtype=int)
    for row in range(len(distances)):
        voter.y_train = labels[row]
        predictions[row] = voter._predict_labels(distances[row:row + 1])[0]
    return predictions


def run_dimension_experiment(training_sizes=(1000, 10000, 100000, 1000000),
                             dimensions=(2, 10, 50, 100, 500), validation_size=1000,
                             seeds=(100, 101, 102), k=3, batch_size=5000,
                             query_batch_size=256):
    """Search every training point in batches; store only nearest neighbors and sums."""
    sizes, dims = np.asarray(training_sizes), np.asarray(dimensions)
    if np.any(np.diff(sizes) <= 0) or sizes[0] < k or np.any(dims < 2):
        raise ValueError("Training sizes must increase and be at least k; dimensions must be at least 2.")
    accuracy = np.zeros((len(seeds), len(sizes), len(dims)))
    ratios = np.zeros_like(accuracy)
    label_agreement = np.zeros_like(accuracy)
    started = perf_counter()

    for repetition, seed in enumerate(seeds):
        train_rng = np.random.default_rng(seed)
        val_rng = np.random.default_rng(seed + 10000)
        validation = val_rng.uniform(-1, 1, size=(validation_size, int(dims.max())))
        best_distances = {d: np.full((validation_size, k), np.inf) for d in dims}
        best_labels = {d: np.zeros((validation_size, k), dtype=int) for d in dims}
        distance_sums = {d: np.zeros(validation_size) for d in dims}

        seen = 0
        for size_index, size in enumerate(sizes):
            while seen < size:
                count = min(batch_size, int(size - seen))
                block = train_rng.uniform(-1, 1, size=(count, int(dims.max())))
                for dimension in dims:
                    features = block[:, :dimension]
                    labels = (features.sum(axis=1) > 0).astype(int)
                    model = KNNClassifier(k=k, vectorized=True)
                    model.train(features, labels)
                    for start in range(0, validation_size, query_batch_size):
                        stop = min(start + query_batch_size, validation_size)
                        with np.errstate(all="ignore"):
                            distances = model._compute_distances_vectorized(validation[start:stop, :dimension])
                        selection = slice(start, stop)
                        best_distances[dimension][selection], best_labels[dimension][selection] = _merge_neighbors(
                            best_distances[dimension][selection], best_labels[dimension][selection],
                            distances, labels, k,
                        )
                        distance_sums[dimension][selection] += distances.sum(axis=1)
                seen += count

            for dim_index, dimension in enumerate(dims):
                prediction = _vote(best_distances[dimension], best_labels[dimension], k)
                truth = (validation[:, :dimension].sum(axis=1) > 0).astype(int)
                accuracy[repetition, size_index, dim_index] = np.mean(prediction == truth)
                ratios[repetition, size_index, dim_index] = np.mean(
                    best_distances[dimension][:, 0] / (distance_sums[dimension] / seen)
                )
                label_agreement[repetition, size_index, dim_index] = np.mean(
                    best_labels[dimension] == truth[:, None]
                )
            print(f"Repeat {repetition + 1}/{len(seeds)}: searched {seen:,} training points "
                  f"at {len(dims)} feature counts ({perf_counter() - started:.0f} s)", flush=True)

    size_labels = [f"{n // 1000000}M" if n >= 1000000 else f"{n // 1000}k" if n >= 1000 else str(n) for n in sizes]
    colors = [SIZE_COLORS[i % len(SIZE_COLORS)] for i in range(len(sizes))]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for index, (size_label, color) in enumerate(zip(size_labels, colors)):
        mean = accuracy[:, index].mean(axis=0)
        spread = accuracy[:, index].std(axis=0)
        ax.plot(dims, mean, "o-", color=color, label=f"{size_label} training points")
        ax.fill_between(dims, mean - spread, mean + spread, color=color, alpha=0.12)
    ax.set(xscale="log", xlabel="Number of features", ylabel="Validation accuracy",
           ylim=(0, 1.03), title="More training data helps, but high dimensions remain difficult")
    ax.set_xticks(dims, labels=dims)
    ax.axhline(0.5, color="0.5", linestyle="--", label="Random guessing")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()

    # Explain the accuracy drop with two quantities on the same axes as the accuracy plot.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    fig.suptitle("Why accuracy drops: neighbors stop being near, so their labels stop being informative")
    panels = [
        (100 * ratios, "Nearest distance (% of mean distance)",
         "How close is the nearest training point?", 100, "Mean distance to training points"),
        (100 * label_agreement, "Neighbors with the query's label (%)",
         f"Do the {k} nearest neighbors share the query's label?", 50, "Coin flip"),
    ]
    for ax, (values, ylabel, title, reference, reference_label) in zip(axes, panels):
        for index, (size_label, color) in enumerate(zip(size_labels, colors)):
            mean = values[:, index].mean(axis=0)
            ax.plot(dims, mean, "o-", color=color, label=f"{size_label} training points")
        ax.axhline(reference, color="0.5", linestyle="--", label=reference_label)
        ax.set(xscale="log", xlabel="Number of features", ylabel=ylabel, ylim=(0, 103), title=title)
        ax.set_xticks(dims, labels=dims)
        ax.grid(alpha=0.2)
        ax.legend()
    plt.show()
    print(f"Finished in {perf_counter() - started:.1f} seconds. "
          f"Each accuracy uses {validation_size:,} separate validation points per repeat.")
    return {"training_sizes": sizes, "dimensions": dims, "accuracy": accuracy,
            "distance_ratios": ratios, "label_agreement": label_agreement}
