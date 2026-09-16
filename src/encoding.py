"""
Codificación cromosómica para los tres subproblemas de optimización.

Este módulo concentra la representación de los individuos: cómo se crean,
cómo se decodifican a parámetros del problema y cómo se validan. Mantenerlo
separado de los operadores y del fitness permite reutilizar el mismo núcleo
evolutivo en los tres subproblemas.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Codificación binaria  ->  Selección de características
# ---------------------------------------------------------------------------

class BinaryEncoding:
    """Cromosoma binario de longitud n_genes.

    Cada gen indica si la variable candidata correspondiente está activa (1)
    o inactiva (0) en el subconjunto evaluado.

    Ejemplo: con variables [u, g, r, i, z, redshift] el cromosoma
    [1, 0, 1, 0, 0, 1] representa el subconjunto {u, r, redshift}.
    """

    def __init__(self, n_genes, rng=None):
        if n_genes < 1:
            raise ValueError("n_genes debe ser >= 1")
        self.n_genes = n_genes
        self.rng = rng if rng is not None else np.random.default_rng()

    def random_individual(self):
        """Genera un individuo aleatorio garantizando al menos un gen activo."""
        ind = self.rng.integers(0, 2, size=self.n_genes)
        if ind.sum() == 0:
            ind[self.rng.integers(0, self.n_genes)] = 1
        return ind

    def random_population(self, pop_size):
        return np.array([self.random_individual() for _ in range(pop_size)])

    def repair(self, chromosome):
        """Corrige cromosomas inválidos (todos los genes en 0)."""
        if chromosome.sum() == 0:
            chromosome[self.rng.integers(0, self.n_genes)] = 1
        return chromosome

    def decode(self, chromosome, feature_names):
        """Devuelve la lista de nombres de variables activas."""
        return [name for name, gen in zip(feature_names, chromosome) if gen == 1]

    def is_valid(self, chromosome):
        return (len(chromosome) == self.n_genes
                and set(np.unique(chromosome)).issubset({0, 1})
                and chromosome.sum() >= 1)


# ---------------------------------------------------------------------------
# 2. Codificación real acotada  ->  Hiperparámetros y centroides
# ---------------------------------------------------------------------------

class RealEncoding:
    """Cromosoma de valores reales con límites por gen.

    Se usa en dos contextos:
      - Hiperparámetros: un gen por hiperparámetro (p. ej. k de vecinos).
      - Clustering: k * n_features genes, que se decodifican como una matriz
        de k centroides en el espacio de las magnitudes fotométricas.
    """

    def __init__(self, lower_bounds, upper_bounds, rng=None):
        self.lower = np.asarray(lower_bounds, dtype=float)
        self.upper = np.asarray(upper_bounds, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower_bounds y upper_bounds deben tener igual longitud")
        if np.any(self.lower > self.upper):
            raise ValueError("cada lower_bound debe ser <= su upper_bound")
        self.n_genes = len(self.lower)
        self.rng = rng if rng is not None else np.random.default_rng()

    def random_individual(self):
        return self.rng.uniform(self.lower, self.upper)

    def random_population(self, pop_size):
        return self.rng.uniform(self.lower, self.upper,
                                size=(pop_size, self.n_genes))

    def repair(self, chromosome):
        """Recorta los genes al rango permitido."""
        return np.clip(chromosome, self.lower, self.upper)

    def is_valid(self, chromosome):
        return (len(chromosome) == self.n_genes
                and np.all(chromosome >= self.lower)
                and np.all(chromosome <= self.upper))

    # -- decodificadores específicos -------------------------------------

    @staticmethod
    def decode_k_neighbors(chromosome):
        """Decodifica el primer gen como un k entero >= 1."""
        return max(1, int(round(float(chromosome[0]))))

    @staticmethod
    def decode_centroids(chromosome, k, n_features):
        """Reorganiza el vector plano en una matriz (k, n_features)."""
        expected = k * n_features
        if len(chromosome) != expected:
            raise ValueError(
                f"El cromosoma debe tener {expected} genes, tiene {len(chromosome)}")
        return np.asarray(chromosome, dtype=float).reshape(k, n_features)

    @classmethod
    def for_centroids(cls, X, k, rng=None):
        """Construye la codificación con límites tomados del rango de los datos."""
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        return cls(np.tile(min_vals, k), np.tile(max_vals, k), rng=rng)


# ---------------------------------------------------------------------------
# Métrica de diversidad poblacional (común a ambas codificaciones)
# ---------------------------------------------------------------------------

def population_diversity(pop):
    """Diversidad como promedio de la desviación estándar por gen.

    Un valor cercano a 0 indica que la población convergió (todos los
    individuos son casi idénticos), lo que suele anticipar el estancamiento
    del algoritmo.
    """
    pop = np.asarray(pop, dtype=float)
    if pop.ndim != 2 or pop.shape[0] < 2:
        return 0.0
    return float(np.mean(np.std(pop, axis=0)))
