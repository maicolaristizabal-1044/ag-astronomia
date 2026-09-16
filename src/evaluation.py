"""
Evaluación: registro del historial evolutivo, métricas y gráficas.

Separar esta capa del núcleo evolutivo permite que los tres subproblemas
generen exactamente el mismo formato de logs y de curvas de convergencia.
"""

import os
import csv

import matplotlib
matplotlib.use("Agg")  # backend sin ventana, necesario dentro del contenedor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Historial evolutivo
# ---------------------------------------------------------------------------

class EvolutionHistory:
    """Acumula, por generación, el mejor fitness, el promedio y la diversidad."""

    def __init__(self, name):
        self.name = name
        self.generation = []
        self.best_fitness = []
        self.mean_fitness = []
        self.diversity = []

    def record(self, gen, fitnesses, diversity):
        fits = np.asarray(fitnesses, dtype=float)
        self.generation.append(int(gen))
        self.best_fitness.append(float(np.max(fits)))
        self.mean_fitness.append(float(np.mean(fits)))
        self.diversity.append(float(diversity))

    def to_csv(self, path):
        ensure_dir(os.path.dirname(path))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["generation", "best_fitness",
                             "mean_fitness", "diversity"])
            for row in zip(self.generation, self.best_fitness,
                           self.mean_fitness, self.diversity):
                writer.writerow(row)
        return path


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def plot_convergence(history, path, title, ylabel):
    """Curva de convergencia: mejor individuo y promedio por generación,
    con la diversidad poblacional en un eje secundario."""
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(history.generation, history.best_fitness,
            marker="o", label="Mejor individuo")
    ax.plot(history.generation, history.mean_fitness,
            marker="s", linestyle="--", label="Fitness promedio")
    ax.set_xlabel("Generación")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(history.generation, history.diversity,
             color="gray", alpha=0.6, linestyle=":", label="Diversidad")
    ax2.set_ylabel("Diversidad poblacional")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrix(y_true, y_pred, labels, path, title):
    ensure_dir(os.path.dirname(path))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm.tolist()


def plot_clustering_comparison(X_2d, ag_labels, kmeans_labels, true_labels,
                               path, explained_variance=None):
    """Tres paneles sobre la misma proyección PCA: AG, KMeans y clases reales.

    Las clases reales se dibujan con una leyenda porque son categorías con
    nombre, mientras que los clusters son índices arbitrarios.
    """
    ensure_dir(os.path.dirname(path))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=ag_labels,
                    cmap="viridis", alpha=0.6, s=18)
    axes[0].set_title("Clusters del AG (3 centroides)")

    axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels,
                    cmap="viridis", alpha=0.6, s=18)
    axes[1].set_title("KMeans clásico (k=3)")

    for cls in sorted(set(true_labels)):
        mask = np.asarray(true_labels) == cls
        axes[2].scatter(X_2d[mask, 0], X_2d[mask, 1], alpha=0.6, s=18, label=cls)
    axes[2].set_title("Clases reales del dataset")
    axes[2].legend(fontsize=9)

    xlabel, ylabel = "PC1", "PC2"
    if explained_variance is not None:
        xlabel = f"PC1 ({explained_variance[0] * 100:.1f}% var.)"
        ylabel = f"PC2 ({explained_variance[1] * 100:.1f}% var.)"
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel(ylabel)

    fig.suptitle("Comparación de agrupamientos sobre magnitudes fotométricas")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
