"""Pruebas de la codificación cromosómica de los tres subproblemas."""

import numpy as np
import pytest

from src.encoding import BinaryEncoding, RealEncoding, population_diversity

SEED = 123


# ---------------------------------------------------------------------------
# Codificación binaria (selección de características)
# ---------------------------------------------------------------------------

@pytest.fixture
def binario():
    return BinaryEncoding(6, rng=np.random.default_rng(SEED))


def test_cromosoma_binario_tiene_la_longitud_correcta(binario):
    ind = binario.random_individual()
    assert len(ind) == 6


def test_cromosoma_binario_solo_contiene_ceros_y_unos(binario):
    pop = binario.random_population(30)
    assert set(np.unique(pop)).issubset({0, 1})


def test_cromosoma_binario_nunca_queda_sin_variables_activas(binario):
    for _ in range(50):
        assert binario.random_individual().sum() >= 1


def test_repair_activa_un_gen_si_el_cromosoma_es_todo_ceros(binario):
    reparado = binario.repair(np.zeros(6, dtype=int))
    assert reparado.sum() == 1


def test_decode_devuelve_los_nombres_de_las_variables_activas(binario):
    nombres = ["u", "g", "r", "i", "z", "redshift"]
    assert binario.decode(np.array([1, 0, 1, 0, 0, 1]), nombres) == ["u", "r", "redshift"]


def test_is_valid_rechaza_cromosomas_invalidos(binario):
    assert binario.is_valid(np.array([1, 0, 1, 0, 0, 1]))
    assert not binario.is_valid(np.zeros(6, dtype=int))     # sin genes activos
    assert not binario.is_valid(np.array([1, 0, 1]))        # longitud incorrecta
    assert not binario.is_valid(np.array([2, 0, 1, 0, 0, 1]))  # valor no binario


def test_poblacion_binaria_tiene_la_forma_esperada(binario):
    assert binario.random_population(20).shape == (20, 6)


def test_encoding_binario_rechaza_longitud_invalida():
    with pytest.raises(ValueError):
        BinaryEncoding(0)


# ---------------------------------------------------------------------------
# Codificación real (hiperparámetros y centroides)
# ---------------------------------------------------------------------------

def test_cromosoma_real_respeta_los_limites():
    enc = RealEncoding([1], [50], rng=np.random.default_rng(SEED))
    pop = enc.random_population(100)
    assert np.all(pop >= 1) and np.all(pop <= 50)


def test_repair_recorta_genes_fuera_de_rango():
    enc = RealEncoding([1, 1], [10, 10])
    reparado = enc.repair(np.array([-5.0, 99.0]))
    assert np.allclose(reparado, [1.0, 10.0])


def test_decode_k_neighbors_devuelve_entero_mayor_o_igual_a_uno():
    assert RealEncoding.decode_k_neighbors(np.array([7.4])) == 7
    assert RealEncoding.decode_k_neighbors(np.array([7.6])) == 8
    assert RealEncoding.decode_k_neighbors(np.array([0.2])) == 1


def test_decode_centroids_devuelve_matriz_k_por_features():
    cromosoma = np.arange(15, dtype=float)   # 3 centroides x 5 features
    centroides = RealEncoding.decode_centroids(cromosoma, k=3, n_features=5)
    assert centroides.shape == (3, 5)
    assert np.allclose(centroides[0], [0, 1, 2, 3, 4])


def test_decode_centroids_falla_si_la_longitud_no_coincide():
    with pytest.raises(ValueError):
        RealEncoding.decode_centroids(np.arange(10, dtype=float), k=3, n_features=5)


def test_for_centroids_construye_limites_desde_los_datos():
    X = np.array([[1.0, 10.0], [3.0, 20.0]])
    enc = RealEncoding.for_centroids(X, k=3)
    assert enc.n_genes == 6                      # 3 centroides x 2 features
    assert np.allclose(enc.lower, [1, 10, 1, 10, 1, 10])
    assert np.allclose(enc.upper, [3, 20, 3, 20, 3, 20])


def test_encoding_real_rechaza_limites_inconsistentes():
    with pytest.raises(ValueError):
        RealEncoding([10], [1])
    with pytest.raises(ValueError):
        RealEncoding([1, 2], [10])


# ---------------------------------------------------------------------------
# Diversidad poblacional
# ---------------------------------------------------------------------------

def test_diversidad_es_cero_si_todos_los_individuos_son_iguales():
    pop = np.ones((10, 5))
    assert population_diversity(pop) == pytest.approx(0.0)


def test_diversidad_es_positiva_en_una_poblacion_variada():
    rng = np.random.default_rng(SEED)
    assert population_diversity(rng.uniform(0, 10, size=(20, 5))) > 0
