"""
Funciones de fitness de los tres subproblemas.

Cada función traduce un cromosoma a un número real donde MAYOR es MEJOR,
que es la convención que asumen los operadores de selección.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

EPS = 1e-9


# ---------------------------------------------------------------------------
# 1. Selección de características  ->  accuracy de KNN (k=5)
# ---------------------------------------------------------------------------

def feature_selection_fitness(chromosome, feature_names,
                              X_train, X_test, y_train, y_test,
                              n_neighbors=5):
    """Accuracy de un KNN entrenado solo con las variables activas.

    Un cromosoma sin variables activas no define un modelo, por lo que recibe
    fitness 0 y queda descartado por la selección.
    """
    active = [name for name, gen in zip(feature_names, chromosome) if gen == 1]
    if not active:
        return 0.0
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train[active], y_train)
    preds = model.predict(X_test[active])
    return float(accuracy_score(y_test, preds))


# ---------------------------------------------------------------------------
# 2. Hiperparámetros  ->  inverso del MSE
# ---------------------------------------------------------------------------

def mse_to_fitness(mse):
    """Convierte un error en fitness: menor MSE => mayor fitness."""
    return 1.0 / (float(mse) + EPS)


def hyperparameter_fitness(chromosome, X_train, X_test, y_train, y_test):
    """Fitness = 1 / MSE de un KNN-regresor con k decodificado del cromosoma.

    Devuelve (fitness, mse) para no tener que reentrenar el modelo al
    reportar métricas.
    """
    k = max(1, int(round(float(chromosome[0]))))
    k = min(k, len(X_train))  # k no puede superar el número de muestras
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse_to_fitness(mse), float(mse)


# ---------------------------------------------------------------------------
# 3. Clustering  ->  inverso del SSE intra-cluster
# ---------------------------------------------------------------------------

def compute_sse(X, centroids):
    """Suma de errores cuadráticos: cada punto aporta la distancia al
    centroide más cercano, elevada al cuadrado."""
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :],
                               axis=2)
    return float(np.sum(np.min(distances, axis=1) ** 2))


def assign_labels(X, centroids):
    """Etiqueta cada punto con el índice de su centroide más cercano."""
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :],
                               axis=2)
    return np.argmin(distances, axis=1)


def clustering_fitness(chromosome, X, k, n_features):
    """Fitness = 1 / SSE. Particiones más compactas obtienen mayor fitness.

    Devuelve (fitness, sse).
    """
    centroids = np.asarray(chromosome, dtype=float).reshape(k, n_features)
    sse = compute_sse(X, centroids)
    return 1.0 / (sse + EPS), sse
