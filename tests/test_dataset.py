"""Pruebas básicas de integridad del dataset astronómico."""

import os

import pandas as pd
import pytest

DATA_PATH = os.environ.get("AG_DATA_PATH", "data/sdss_sample.csv")
COLUMNAS_REQUERIDAS = ["u", "g", "r", "i", "z", "redshift", "class"]


@pytest.fixture(scope="module")
def df():
    assert os.path.exists(DATA_PATH), f"No se encontró el dataset en {DATA_PATH}"
    return pd.read_csv(DATA_PATH)


def test_dataset_no_esta_vacio(df):
    assert len(df) > 0, "El dataset no tiene filas"


def test_columnas_requeridas_presentes(df):
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    assert not faltantes, f"Faltan columnas en el dataset: {faltantes}"


def test_magnitudes_son_numericas(df):
    for col in ["u", "g", "r", "i", "z", "redshift"]:
        assert pd.api.types.is_numeric_dtype(df[col]), \
            f"La columna '{col}' debería ser numérica"


def test_sin_valores_nulos_en_columnas_usadas(df):
    nulos = df[COLUMNAS_REQUERIDAS].isnull().sum()
    assert nulos.sum() == 0, f"Hay valores nulos:\n{nulos[nulos > 0]}"


def test_variable_objetivo_tiene_al_menos_dos_clases(df):
    n_clases = df["class"].nunique()
    assert n_clases >= 2, f"Se esperaban al menos 2 clases, hay {n_clases}"


def test_hay_suficientes_muestras_para_particion_70_30(df):
    # Con un 30% de test y KNN k=5 se necesita un mínimo razonable de filas
    assert len(df) >= 50, "Muy pocas filas para una partición 70/30 con KNN k=5"


def test_magnitudes_en_rango_fisicamente_plausible(df):
    for col in ["u", "g", "r", "i", "z"]:
        assert df[col].between(0, 40).all(), \
            f"La columna '{col}' tiene magnitudes fuera del rango esperado [0, 40]"
