"""
Subproblema 2: ajuste de hiperparámetros con Algoritmo Genético.

Variables de entrada : u, g, r, i, z
Variable objetivo    : redshift
Modelo base          : KNN-regresor
Codificación         : cromosoma real de 1 gen (k de vecinos, rango [1, 50])
Fitness              : 1 / MSE  (favorece configuraciones con menor error)
Operadores           : ruleta, cruce aritmético, mutación gaussiana, elitismo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

from .encoding import RealEncoding, population_diversity
from .operators import (roulette_selection, arithmetic_crossover,
                        gaussian_mutation, elitism)
from .fitness import hyperparameter_fitness
from .evaluation import EvolutionHistory, plot_convergence

INPUT_FEATURES = ["u", "g", "r", "i", "z"]
TARGET = "redshift"


class HyperparameterAG:

    def __init__(self, data_path, pop_size=20, generations=20,
                 mutation_rate=0.2, sigma=2.0, crossover_rate=0.9,
                 n_elite=1, k_min=1, k_max=50,
                 test_size=0.3, random_state=42):
        self.df = pd.read_csv(data_path)
        self.X = self.df[INPUT_FEATURES]
        self.y = self.df[TARGET]

        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.sigma = sigma
        self.crossover_rate = crossover_rate
        self.n_elite = n_elite
        self.k_min, self.k_max = k_min, k_max
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)
        self.encoder = RealEncoding([k_min], [k_max], rng=self.rng)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    # -- fitness ----------------------------------------------------------

    def fitness(self, chromosome):
        """Devuelve (fitness, mse)."""
        return hyperparameter_fitness(
            chromosome, self.X_train, self.X_test, self.y_train, self.y_test)

    # -- bucle evolutivo --------------------------------------------------

    def run(self, output_dir="outputs"):
        pop = self.encoder.random_population(self.pop_size)
        history = EvolutionHistory("hyperparameter_tuning")
        best_chromosome, best_fitness = None, -np.inf

        for gen in range(self.generations):
            evals = [self.fitness(ind) for ind in pop]
            fits = np.array([e[0] for e in evals])
            history.record(gen, fits, population_diversity(pop))

            gen_best = int(np.argmax(fits))
            if fits[gen_best] > best_fitness:
                best_fitness = float(fits[gen_best])
                best_chromosome = pop[gen_best].copy()

            new_pop = elitism(pop, fits, self.n_elite)

            while len(new_pop) < self.pop_size:
                p1 = roulette_selection(pop, fits, self.rng)
                p2 = roulette_selection(pop, fits, self.rng)
                c1, c2 = arithmetic_crossover(p1, p2, self.rng, self.crossover_rate)
                for child in (c1, c2):
                    if len(new_pop) < self.pop_size:
                        child = gaussian_mutation(
                            child, self.rng, self.mutation_rate, self.sigma,
                            lower=self.encoder.lower, upper=self.encoder.upper)
                        new_pop.append(self.encoder.repair(child))

            pop = np.array(new_pop)

        # Métricas del mejor individuo
        best_k = RealEncoding.decode_k_neighbors(best_chromosome)
        model = KNeighborsRegressor(n_neighbors=best_k)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        mse = float(mean_squared_error(self.y_test, preds))
        r2 = float(r2_score(self.y_test, preds))

        # Línea base para comparar: k=5, el valor por defecto habitual
        base = KNeighborsRegressor(n_neighbors=5)
        base.fit(self.X_train, self.y_train)
        base_preds = base.predict(self.X_test)

        plot_convergence(
            history, f"{output_dir}/ht_fitness_curve.png",
            "Evolución del fitness - Ajuste de hiperparámetros",
            "Fitness (1 / MSE)",
        )
        history.to_csv(f"{output_dir}/ht_fitness_history.csv")

        return {
            "best_k": int(best_k),
            "best_chromosome": [float(g) for g in best_chromosome],
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2_score": r2,
            "best_fitness": best_fitness,
            "baseline_k5": {
                "mse": float(mean_squared_error(self.y_test, base_preds)),
                "r2_score": float(r2_score(self.y_test, base_preds)),
            },
            "params": {
                "model": "KNeighborsRegressor",
                "pop_size": self.pop_size,
                "generations": self.generations,
                "mutation_rate": self.mutation_rate,
                "sigma": self.sigma,
                "crossover_rate": self.crossover_rate,
                "n_elite": self.n_elite,
                "k_range": [self.k_min, self.k_max],
                "test_size": 0.3,
                "random_state": self.random_state,
            },
        }
