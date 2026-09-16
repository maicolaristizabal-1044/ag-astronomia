"""
Subproblema 1: selección de variables relevantes con Algoritmo Genético.

Variables candidatas : u, g, r, i, z, redshift
Variable objetivo    : class
Codificación         : cromosoma binario (1 = variable activa)
Fitness              : accuracy de KNN con k=5, partición 70/30
Operadores           : torneo, cruce en un punto, mutación bit-flip
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from .encoding import BinaryEncoding, population_diversity
from .operators import (tournament_selection, one_point_crossover,
                        bit_flip_mutation, elitism)
from .fitness import feature_selection_fitness
from .evaluation import (EvolutionHistory, plot_convergence,
                         plot_confusion_matrix)

CANDIDATE_FEATURES = ["u", "g", "r", "i", "z", "redshift"]
TARGET = "class"


class FeatureSelectionAG:

    def __init__(self, data_path, pop_size=20, generations=15,
                 mutation_rate=0.1, crossover_rate=0.9, tournament_size=3,
                 n_elite=2, n_neighbors=5, test_size=0.3, random_state=42):
        self.df = pd.read_csv(data_path)
        self.feature_names = CANDIDATE_FEATURES
        self.X = self.df[self.feature_names]
        self.y = self.df[TARGET]

        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.n_elite = n_elite
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)
        self.encoder = BinaryEncoding(len(self.feature_names), rng=self.rng)

        # Partición 70/30 estratificada: conserva la proporción de clases
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,
        )

    # -- fitness ----------------------------------------------------------

    def fitness(self, chromosome):
        return feature_selection_fitness(
            chromosome, self.feature_names,
            self.X_train, self.X_test, self.y_train, self.y_test,
            n_neighbors=self.n_neighbors,
        )

    def _evaluate_population(self, pop):
        return np.array([self.fitness(ind) for ind in pop])

    # -- bucle evolutivo --------------------------------------------------

    def run(self, output_dir="outputs"):
        pop = self.encoder.random_population(self.pop_size)
        history = EvolutionHistory("feature_selection")
        best_chromosome, best_fitness = None, -np.inf

        for gen in range(self.generations):
            fits = self._evaluate_population(pop)
            history.record(gen, fits, population_diversity(pop))

            gen_best = int(np.argmax(fits))
            if fits[gen_best] > best_fitness:
                best_fitness = float(fits[gen_best])
                best_chromosome = pop[gen_best].copy()

            # Elitismo: los mejores pasan intactos a la siguiente generación
            new_pop = elitism(pop, fits, self.n_elite)

            while len(new_pop) < self.pop_size:
                p1 = tournament_selection(pop, fits, self.rng, self.tournament_size)
                p2 = tournament_selection(pop, fits, self.rng, self.tournament_size)
                c1, c2 = one_point_crossover(p1, p2, self.rng, self.crossover_rate)
                for child in (c1, c2):
                    if len(new_pop) < self.pop_size:
                        child = bit_flip_mutation(child, self.rng, self.mutation_rate)
                        new_pop.append(self.encoder.repair(child))

            pop = np.array(new_pop)

        # Evaluación final del mejor individuo encontrado
        selected = self.encoder.decode(best_chromosome, self.feature_names)
        model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        model.fit(self.X_train[selected], self.y_train)
        preds = model.predict(self.X_test[selected])
        accuracy = float(accuracy_score(self.y_test, preds))
        labels = sorted(self.y.unique())

        # Línea base: todas las variables activas, para poder comparar
        baseline = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        baseline.fit(self.X_train, self.y_train)
        baseline_acc = float(accuracy_score(self.y_test, baseline.predict(self.X_test)))

        cm = plot_confusion_matrix(
            self.y_test, preds, labels,
            f"{output_dir}/fs_confusion_matrix.png",
            "Matriz de confusión - Mejor subconjunto del AG",
        )
        plot_convergence(
            history, f"{output_dir}/fs_fitness_curve.png",
            "Evolución del fitness - Selección de características",
            "Accuracy",
        )
        history.to_csv(f"{output_dir}/fs_fitness_history.csv")

        return {
            "best_chromosome": [int(g) for g in best_chromosome],
            "selected_features": selected,
            "n_features_selected": len(selected),
            "best_accuracy": accuracy,
            "baseline_all_features_accuracy": baseline_acc,
            "confusion_matrix": cm,
            "confusion_matrix_labels": labels,
            "classification_report": classification_report(
                self.y_test, preds, output_dict=True, zero_division=0),
            "params": {
                "pop_size": self.pop_size,
                "generations": self.generations,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "tournament_size": self.tournament_size,
                "n_elite": self.n_elite,
                "knn_k": self.n_neighbors,
                "test_size": 0.3,
                "random_state": self.random_state,
            },
        }
