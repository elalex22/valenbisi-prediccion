import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from valenbisi.features import FEATURES
from valenbisi.modelo import GRUPOS, dataset, entrenar, importancia_permutacion, matriz

COLUMNAS = ["senal", "ruido", "estacion"]


def datos(n: int, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    """``senal`` decide la probabilidad (5 % o 40 %); ``ruido`` no aporta nada."""
    rng = np.random.default_rng(semilla)
    senal = rng.integers(0, 2, n)
    y = rng.random(n) < np.where(senal == 1, 0.4, 0.05)
    X = np.column_stack([senal, rng.random(n), rng.integers(1, 5, n)]).astype("float32")
    return X, y


def test_matriz_conserva_nulos_y_pasa_booleanos_a_numeros():
    tabla = pa.Table.from_pandas(pd.DataFrame({"a": [True, False], "b": [1.5, np.nan]}), preserve_index=False)
    X = matriz(tabla, ["a", "b"])

    assert X.dtype == np.float32 and X.flags["C_CONTIGUOUS"]
    assert X[:, 0].tolist() == [1.0, 0.0]
    assert X[0, 1] == 1.5 and np.isnan(X[1, 1])


def test_grupos_cubren_cada_feature_una_vez():
    en_grupos = [c for columnas in GRUPOS.values() for c in columnas]
    assert sorted(en_grupos) == sorted(FEATURES)


def test_sin_pesos_las_probabilidades_salen_calibradas():
    X, y = datos(40_000, 0)
    X_val, y_val = datos(20_000, 1)
    entrenamiento = dataset(X, y, COLUMNAS)
    validacion = dataset(X_val, y_val, COLUMNAS, referencia=entrenamiento)
    parametros = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 100,
                  "num_threads": 2, "seed": 0, "verbose": -1}
    modelo = entrenar(entrenamiento, validacion, parametros, max_arboles=200, paciencia=20, registro=0)
    p = modelo.predict(X_val, num_iteration=modelo.best_iteration)

    assert modelo.best_iteration < 200  # la parada temprana actuó
    for valor, esperada in ((0, 0.05), (1, 0.4)):
        assert p[X_val[:, 0] == valor].mean() == pytest.approx(esperada, abs=0.02)
    assert p.mean() == pytest.approx(y_val.mean(), abs=0.01)

    _, caidas = importancia_permutacion(modelo.predict, X_val, y_val, {"senal": [0], "ruido": [1]})
    assert caidas["senal"] > 0.1
    assert abs(caidas["ruido"]) < 0.02
