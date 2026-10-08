from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def crps_map(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pointwise ensemble CRPS for samples [N,S,C,T], target [N,C,T]."""
    values = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if values.ndim != 4 or truth.shape != (
        values.shape[0],
        values.shape[2],
        values.shape[3],
    ):
        raise ValueError("Expected samples [N,S,C,T] and target [N,C,T].")
    term1 = np.mean(np.abs(values - truth[:, None]), axis=1)
    ordered = np.sort(values, axis=1)
    size = values.shape[1]
    coefficients = (2 * np.arange(size) - size + 1).reshape(1, size, 1, 1)
    mean_pairwise = 2.0 * np.sum(coefficients * ordered, axis=1) / size**2
    return term1 - 0.5 * mean_pairwise


def macro_ncrps(samples: np.ndarray, target: np.ndarray) -> float:
    scores = crps_map(samples, target)
    per_channel = [
        np.sum(scores[:, channel])
        / (np.sum(np.abs(target[:, channel])) + 1e-8)
        for channel in range(target.shape[1])
    ]
    return float(np.mean(per_channel))


def compute_precision_recall(
    generated: np.ndarray, real: np.ndarray, k: int = 5
) -> tuple[float, float]:
    generated_values = np.asarray(generated, dtype=np.float64).reshape(
        len(generated), -1
    )
    real_values = np.asarray(real, dtype=np.float64).reshape(len(real), -1)
    if len(generated_values) < 2 or len(real_values) < 2:
        return float("nan"), float("nan")
    effective_k = min(k, len(generated_values) - 1, len(real_values) - 1)

    def radius(values: np.ndarray) -> np.ndarray:
        neighbours = NearestNeighbors(n_neighbors=effective_k + 1).fit(values)
        distances, _ = neighbours.kneighbors(values)
        return distances[:, effective_k]

    real_radius = radius(real_values)
    generated_radius = radius(generated_values)
    distance, index = NearestNeighbors(n_neighbors=1).fit(real_values).kneighbors(
        generated_values
    )
    precision = np.mean(distance[:, 0] <= real_radius[index[:, 0]])
    distance, index = NearestNeighbors(n_neighbors=1).fit(
        generated_values
    ).kneighbors(real_values)
    recall = np.mean(distance[:, 0] <= generated_radius[index[:, 0]])
    return float(precision), float(recall)


def summarize(
    samples: np.ndarray,
    target: np.ndarray,
    labels: list[str],
    target_mean: np.ndarray,
    target_std: np.ndarray,
    seed: int = 42,
    precision_recall_k: int = 5,
    max_precision_samples: int = 10_000,
) -> dict[str, float]:
    """Use the baseline5 evaluation contract; training CRPS remains unchanged."""
    from baseline6.aligned_metrics import summarize as aligned_summarize

    return aligned_summarize(
        samples, target, labels, target_mean, target_std, seed,
        precision_recall_k, max_precision_samples,
    )


def compute_global_pearson_matrix(total_data: np.ndarray) -> np.ndarray:
    values = np.asarray(total_data, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected [N,C,T], got {values.shape}.")
    flattened = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    return np.corrcoef(flattened, rowvar=False)


def _save_heatmap(
    matrix: np.ndarray,
    title: str,
    save_path: str | Path,
    labels: list[str],
    cmap: str,
    vmin: float,
    vmax: float,
) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def save_global_pearson_plot(
    total_real: np.ndarray, save_path: str | Path, labels: list[str]
) -> Path:
    return _save_heatmap(
        compute_global_pearson_matrix(total_real),
        "Global Pearson Correlation",
        save_path,
        labels,
        "coolwarm",
        -1.0,
        1.0,
    )


def save_global_pearson_comparison(
    total_real: np.ndarray,
    total_generated: np.ndarray,
    output_dir: str | Path,
    labels: list[str],
) -> dict[str, float]:
    real_correlation = compute_global_pearson_matrix(total_real)
    generated = np.asarray(total_generated, dtype=np.float64)
    generated_correlation = compute_global_pearson_matrix(
        generated.reshape(-1, generated.shape[2], generated.shape[3])
    )
    difference = np.abs(generated_correlation - real_correlation)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for matrix, title, filename, cmap, vmin, vmax in (
        (real_correlation, "Real Global Pearson", "real_global_pearson.png", "coolwarm", -1.0, 1.0),
        (generated_correlation, "Generated Global Pearson", "generated_global_pearson.png", "coolwarm", -1.0, 1.0),
        (difference, "Pearson Absolute Error", "pearson_difference.png", "Reds", 0.0, 1.0),
    ):
        _save_heatmap(matrix, title, destination / filename, labels, cmap, vmin, vmax)
    return {
        "Pearson_MAE": float(difference.mean()),
        "Pearson_RMSE": float(np.sqrt(np.mean(difference**2))),
    }


def save_random_timeseries_plots(
    total_real: np.ndarray,
    total_generated: np.ndarray,
    dates: list[str],
    output_dir: str | Path,
    labels: list[str],
    seed: int = 42,
    n_plots: int = 50,
) -> list[Path]:
    import matplotlib.pyplot as plt

    real = np.asarray(total_real)
    generated = np.asarray(total_generated)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected = np.random.default_rng(seed).choice(
        real.shape[0], size=min(n_plots, real.shape[0]), replace=False
    )
    saved: list[Path] = []
    for index in selected:
        figure, axes = plt.subplots(
            len(labels), 1, figsize=(10, 3 * len(labels)), sharex=True
        )
        if len(labels) == 1:
            axes = [axes]
        for channel, label in enumerate(labels):
            for scenario in range(min(100, generated.shape[1])):
                axes[channel].plot(
                    generated[index, scenario, channel],
                    color="red",
                    alpha=0.1,
                    linewidth=1,
                )
            axes[channel].plot(
                real[index, channel],
                color="black",
                linewidth=2,
                linestyle="--",
                label="Ground Truth",
            )
            axes[channel].set_title(label)
            axes[channel].grid(True, alpha=0.3)
            axes[channel].legend()
        figure.suptitle(f"Generated scenarios - {dates[index]}")
        figure.tight_layout()
        save_path = destination / f"{dates[index]}.png"
        figure.savefig(save_path, dpi=200)
        plt.close(figure)
        saved.append(save_path)
    return saved

