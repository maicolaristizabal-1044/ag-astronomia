"""Pruebas de los operadores evolutivos y de las funciones de fitness."""

import numpy as np
import pytest

from src.operators import (tournament_selection, roulette_selection,
                           one_point_crossover, arithmetic_crossover,
                           bit_flip_mutation, gaussian_mutation, elitism)
from src.fitness import compute_sse, assign_labels, mse_to_fitness

SEED = 7


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Selección
# ---------------------------------------------------------------------------

def test_torneo_devuelve_un_individuo_de_la_poblacion(rng):
    pop = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    ganador = tournament_selection(pop, np.array([0.1, 0.9, 0.5, 0.2]), rng)
    assert any(np.array_equal(ganador, ind) for ind in pop)


def test_torneo_prefiere_individuos_con_mayor_fitness(rng):
    pop = np.array([[0], [1], [2], [3]])
    fits = np.array([0.0, 0.0, 0.0, 10.0])
    ganados = sum(int(tournament_selection(pop, fits, rng)[0] == 3)
                  for _ in range(200))
    assert ganados > 100  # el mejor gana siempre que entre al torneo


def test_ruleta_no_falla_si_todos_los_fitness_son_cero(rng):
    pop = np.array([[1.0], [2.0], [3.0]])
    seleccionado = roulette_selection(pop, np.zeros(3), rng)
    assert seleccionado[0] in (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Cruce
# ---------------------------------------------------------------------------

def test_cruce_un_punto_conserva_la_longitud(rng):
    p1 = np.array([1, 1, 1, 1, 1, 1])
    p2 = np.array([0, 0, 0, 0, 0, 0])
    c1, c2 = one_point_crossover(p1, p2, rng, crossover_rate=1.0)
    assert len(c1) == len(p1) and len(c2) == len(p2)


def test_cruce_un_punto_mezcla_genes_de_ambos_padres(rng):
    p1 = np.ones(8, dtype=int)
    p2 = np.zeros(8, dtype=int)
    c1, _ = one_point_crossover(p1, p2, rng, crossover_rate=1.0)
    assert 0 < c1.sum() < 8  # tomó genes de los dos padres


def test_cruce_aritmetico_produce_hijos_dentro_del_rango_de_los_padres(rng):
    p1 = np.array([0.0, 0.0])
    p2 = np.array([10.0, 10.0])
    c1, c2 = arithmetic_crossover(p1, p2, rng, crossover_rate=1.0)
    assert np.all(c1 >= 0) and np.all(c1 <= 10)
    assert np.all(c2 >= 0) and np.all(c2 <= 10)


def test_cruce_aritmetico_los_hijos_son_complementarios(rng):
    p1 = np.array([0.0, 2.0])
    p2 = np.array([10.0, 4.0])
    c1, c2 = arithmetic_crossover(p1, p2, rng, crossover_rate=1.0)
    assert np.allclose(c1 + c2, p1 + p2)


def test_sin_cruce_los_hijos_son_copias_de_los_padres(rng):
    p1, p2 = np.array([1, 0, 1]), np.array([0, 1, 0])
    c1, c2 = one_point_crossover(p1, p2, rng, crossover_rate=0.0)
    assert np.array_equal(c1, p1) and np.array_equal(c2, p2)


# ---------------------------------------------------------------------------
# Mutación
# ---------------------------------------------------------------------------

def test_bit_flip_con_tasa_uno_invierte_todos_los_genes(rng):
    original = np.array([1, 0, 1, 0])
    mutado = bit_flip_mutation(original, rng, mutation_rate=1.0)
    assert np.array_equal(mutado, np.array([0, 1, 0, 1]))


def test_bit_flip_con_tasa_cero_no_cambia_nada(rng):
    original = np.array([1, 0, 1, 0])
    assert np.array_equal(bit_flip_mutation(original, rng, 0.0), original)


def test_bit_flip_no_modifica_el_cromosoma_original(rng):
    original = np.array([1, 0, 1, 0])
    copia = original.copy()
    bit_flip_mutation(original, rng, mutation_rate=1.0)
    assert np.array_equal(original, copia)


def test_mutacion_gaussiana_respeta_los_limites(rng):
    original = np.array([5.0, 5.0, 5.0])
    mutado = gaussian_mutation(original, rng, mutation_rate=1.0, sigma=100.0,
                               lower=np.array([0.0, 0.0, 0.0]),
                               upper=np.array([10.0, 10.0, 10.0]))
    assert np.all(mutado >= 0.0) and np.all(mutado <= 10.0)


# ---------------------------------------------------------------------------
# Elitismo
# ---------------------------------------------------------------------------

def test_elitismo_devuelve_los_mejores_individuos():
    pop = np.array([[1], [2], [3], [4]])
    elite = elitism(pop, np.array([0.1, 0.9, 0.4, 0.8]), n_elite=2)
    assert len(elite) == 2
    assert elite[0][0] == 2 and elite[1][0] == 4


def test_elitismo_con_cero_devuelve_lista_vacia():
    pop = np.array([[1], [2]])
    assert elitism(pop, np.array([0.5, 0.7]), n_elite=0) == []


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def test_menor_mse_produce_mayor_fitness():
    assert mse_to_fitness(0.01) > mse_to_fitness(1.0)


def test_sse_es_cero_si_los_centroides_coinciden_con_los_puntos():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert compute_sse(X, X) == pytest.approx(0.0)


def test_sse_aumenta_al_alejar_los_centroides():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    cerca = np.array([[0.0, 0.0], [1.0, 1.0]])
    lejos = np.array([[10.0, 10.0], [20.0, 20.0]])
    assert compute_sse(X, lejos) > compute_sse(X, cerca)


def test_assign_labels_asigna_el_centroide_mas_cercano():
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    centroides = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert np.array_equal(assign_labels(X, centroides), np.array([0, 1]))
