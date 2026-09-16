"""
Subproblema 3: optimización de agrupamiento con Algoritmo Genético.

Datos         : magnitudes fotométricas u, g, r, i, z
Codificación  : cromosoma real de 3 * 5 = 15 genes (coordenadas de 3 centroides)
Fitness       : 1 / SSE intra-cluster (favorece particiones compactas)
Operadores    : torneo, cruce aritmético, mutación gaussiana, elitismo
Comparación   : clusters del AG vs. KMeans clásico vs. clases reales
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .encoding import RealEncoding, population_diversity
from .operators import (tournament_selection, arithmetic_crossover,
                        gaussian_mutation, elitism)
from .fitness import clustering_fitness, compute_sse, assign_labels
from .evaluation import (EvolutionHistory, plot_convergence,
                         plot_clustering_comparison)

PHOTOMETRIC_FEATURES = ["u", "g", "r", "i", "z"]
TARGET = "class"


class ClusteringAG:

    def __init__(self, data_path, pop_size=60, generations=150, k_clusters=3,
                 mutation_rate=0.25, sigma=1.0, sigma_min=0.02,
                 crossover_rate=0.9, tournament_size=3, n_elite=2,
                 seed_fraction=0.5, random_state=42):
        self.df = pd.read_csv(data_path)
        self.X = self.df[PHOTOMETRIC_FEATURES].values
        self.true_labels = self.df[TARGET].values

        self.k = k_clusters
        self.n_features = self.X.shape[1]
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.n_elite = n_elite
        self.seed_fraction = seed_fraction
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)
        # Los límites de cada gen se toman del rango real de los datos,
        # así ningún centroide inicial cae fuera de la nube de puntos.
        self.encoder = RealEncoding.for_centroids(self.X, self.k, rng=self.rng)

    # -- fitness ----------------------------------------------------------

    def fitness(self, chromosome):
        """Devuelve (fitness, sse)."""
        return clustering_fitness(chromosome, self.X, self.k, self.n_features)

    def _current_sigma(self, gen):
        """Decaimiento lineal de sigma entre self.sigma y self.sigma_min."""
        if self.generations <= 1:
            return self.sigma
        avance = gen / (self.generations - 1)
        return self.sigma + (self.sigma_min - self.sigma) * avance

    def _initial_population(self):
        """Inicialización híbrida.

        Una fracción de la población se genera muestreando k puntos reales del
        dataset como centroides iniciales, y el resto de forma uniforme dentro
        de los límites de los datos. Con 15 genes, una inicialización
        totalmente aleatoria coloca casi siempre los centroides en zonas
        vacías del espacio y el AG queda atrapado en óptimos locales; sembrar
        con puntos reales garantiza que parte de la población arranque dentro
        de la nube de datos sin eliminar la exploración.
        """
        n_semillas = int(self.pop_size * self.seed_fraction)
        individuos = []

        for _ in range(n_semillas):
            idx = self.rng.choice(len(self.X), size=self.k, replace=False)
            individuos.append(self.X[idx].flatten().astype(float))

        for _ in range(self.pop_size - n_semillas):
            individuos.append(self.encoder.random_individual())

        return np.array(individuos)

    # -- bucle evolutivo --------------------------------------------------

    def run(self, output_dir="outputs"):
        pop = self._initial_population()
        history = EvolutionHistory("clustering")
        best_chromosome, best_fitness = None, -np.inf

        for gen in range(self.generations):
            evals = [self.fitness(ind) for ind in pop]
            fits = np.array([e[0] for e in evals])
            history.record(gen, fits, population_diversity(pop))

            gen_best = int(np.argmax(fits))
            if fits[gen_best] > best_fitness:
                best_fitness = float(fits[gen_best])
                best_chromosome = pop[gen_best].copy()

            # Sigma adaptativo: la mutación empieza amplia para explorar el
            # espacio y se reduce con las generaciones para refinar la
            # posición de los centroides sin destruir las buenas soluciones.
            sigma_gen = self._current_sigma(gen)

            new_pop = elitism(pop, fits, self.n_elite)

            while len(new_pop) < self.pop_size:
                p1 = tournament_selection(pop, fits, self.rng, self.tournament_size)
                p2 = tournament_selection(pop, fits, self.rng, self.tournament_size)
                c1, c2 = arithmetic_crossover(p1, p2, self.rng, self.crossover_rate)
                for child in (c1, c2):
                    if len(new_pop) < self.pop_size:
                        child = gaussian_mutation(
                            child, self.rng, self.mutation_rate, sigma_gen,
                            lower=self.encoder.lower, upper=self.encoder.upper)
                        new_pop.append(self.encoder.repair(child))

            pop = np.array(new_pop)

        # -- solución del AG ---------------------------------------------
        centroids = RealEncoding.decode_centroids(
            best_chromosome, self.k, self.n_features)
        ag_labels = assign_labels(self.X, centroids)
        ag_sse = compute_sse(self.X, centroids)

        # -- referencia: KMeans clásico ----------------------------------
        kmeans = KMeans(n_clusters=self.k, n_init=10,
                        random_state=self.random_state).fit(self.X)
        km_sse = compute_sse(self.X, kmeans.cluster_centers_)

        # -- proyección PCA solo para visualizar -------------------------
        pca = PCA(n_components=2, random_state=self.random_state)
        X_2d = pca.fit_transform(self.X)

        plot_clustering_comparison(
            X_2d, ag_labels, kmeans.labels_, self.true_labels,
            f"{output_dir}/clustering_comparison.png",
            explained_variance=pca.explained_variance_ratio_,
        )
        plot_convergence(
            history, f"{output_dir}/cl_fitness_curve.png",
            "Evolución del fitness - Clustering evolutivo",
            "Fitness (1 / SSE)",
        )
        history.to_csv(f"{output_dir}/cl_fitness_history.csv")

        # ARI compara una partición con las clases reales sin asumir que las
        # etiquetas coinciden; el silhouette mide compacidad sin usar la clase.
        return {
            "ag": {
                "sse": ag_sse,
                "silhouette": float(silhouette_score(self.X, ag_labels))
                if len(set(ag_labels)) > 1 else None,
                "adjusted_rand_index": float(
                    adjusted_rand_score(self.true_labels, ag_labels)),
                "centroids": centroids.tolist(),
                "cluster_sizes": np.bincount(
                    ag_labels, minlength=self.k).tolist(),
            },
            "kmeans": {
                "sse": km_sse,
                "silhouette": float(silhouette_score(self.X, kmeans.labels_)),
                "adjusted_rand_index": float(
                    adjusted_rand_score(self.true_labels, kmeans.labels_)),
                "centroids": kmeans.cluster_centers_.tolist(),
                "cluster_sizes": np.bincount(
                    kmeans.labels_, minlength=self.k).tolist(),
            },
            "sse_gap_pct": float((ag_sse - km_sse) / km_sse * 100.0),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "params": {
                "k_clusters": self.k,
                "chromosome_length": self.k * self.n_features,
                "pop_size": self.pop_size,
                "generations": self.generations,
                "mutation_rate": self.mutation_rate,
                "sigma_inicial": self.sigma,
                "sigma_final": self.sigma_min,
                "crossover_rate": self.crossover_rate,
                "tournament_size": self.tournament_size,
                "n_elite": self.n_elite,
                "seed_fraction": self.seed_fraction,
                "random_state": self.random_state,
            },
        }
