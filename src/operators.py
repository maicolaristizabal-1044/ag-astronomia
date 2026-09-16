"""
Operadores evolutivos: selección, cruce y mutación.

Todos los operadores son funciones puras que reciben el generador aleatorio
como argumento, de modo que las ejecuciones son reproducibles fijando una
semilla única en el punto de entrada.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Selección
# ---------------------------------------------------------------------------

def tournament_selection(pop, fitnesses, rng, tournament_size=3):
    """Selección por torneo.

    Se toman `tournament_size` individuos al azar y gana el de mayor fitness.
    Presión selectiva moderada y robusta frente a escalas de fitness muy
    dispares (a diferencia de la ruleta).
    """
    n = len(pop)
    size = min(tournament_size, n)
    idx = rng.choice(n, size=size, replace=False)
    winner = idx[np.argmax(np.asarray(fitnesses)[idx])]
    return np.array(pop[winner], copy=True)


def roulette_selection(pop, fitnesses, rng):
    """Selección proporcional al fitness (ruleta).

    Requiere fitness no negativo. Si la suma es 0 se degrada a selección
    uniforme para evitar una división por cero.
    """
    fits = np.asarray(fitnesses, dtype=float)
    fits = np.clip(fits, 0.0, None)
    total = fits.sum()
    if total <= 0 or not np.isfinite(total):
        idx = rng.integers(0, len(pop))
    else:
        idx = rng.choice(len(pop), p=fits / total)
    return np.array(pop[idx], copy=True)


# ---------------------------------------------------------------------------
# Cruce
# ---------------------------------------------------------------------------

def one_point_crossover(p1, p2, rng, crossover_rate=0.9):
    """Cruce en un punto para cromosomas binarios."""
    if rng.random() > crossover_rate or len(p1) < 2:
        return np.array(p1, copy=True), np.array(p2, copy=True)
    pt = int(rng.integers(1, len(p1)))
    c1 = np.concatenate([p1[:pt], p2[pt:]])
    c2 = np.concatenate([p2[:pt], p1[pt:]])
    return c1, c2


def arithmetic_crossover(p1, p2, rng, crossover_rate=0.9):
    """Cruce aritmético para cromosomas reales.

    Genera dos hijos complementarios con un alpha aleatorio por cruce.
    Usar alpha aleatorio (y no 0.5 fijo) evita que la población colapse al
    promedio en pocas generaciones.
    """
    if rng.random() > crossover_rate:
        return np.array(p1, copy=True), np.array(p2, copy=True)
    alpha = rng.random()
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = (1 - alpha) * p1 + alpha * p2
    return c1, c2


# ---------------------------------------------------------------------------
# Mutación
# ---------------------------------------------------------------------------

def bit_flip_mutation(chromosome, rng, mutation_rate=0.1):
    """Mutación bit-flip: cada gen se invierte con probabilidad mutation_rate."""
    mutant = np.array(chromosome, copy=True)
    mask = rng.random(len(mutant)) < mutation_rate
    mutant[mask] = 1 - mutant[mask]
    return mutant


def gaussian_mutation(chromosome, rng, mutation_rate=0.2, sigma=1.0,
                      lower=None, upper=None):
    """Mutación gaussiana: suma ruido N(0, sigma) gen a gen.

    Si se entregan límites, el resultado se recorta al rango válido.
    """
    mutant = np.array(chromosome, dtype=float, copy=True)
    mask = rng.random(len(mutant)) < mutation_rate
    if np.any(mask):
        mutant[mask] += rng.normal(0.0, sigma, size=int(mask.sum()))
    if lower is not None and upper is not None:
        mutant = np.clip(mutant, lower, upper)
    return mutant


# ---------------------------------------------------------------------------
# Elitismo
# ---------------------------------------------------------------------------

def elitism(pop, fitnesses, n_elite=1):
    """Devuelve copias de los n_elite mejores individuos de la población."""
    if n_elite <= 0:
        return []
    order = np.argsort(np.asarray(fitnesses))[::-1]
    return [np.array(pop[i], copy=True) for i in order[:n_elite]]
