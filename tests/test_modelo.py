import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from valenbisi.features import FEATURES
from valenbisi.modelo import (
    FEATURES_MODELO,
    GRUPOS,
    GRUPOS_MODELO,
    SecuenciaParquet,
    dataset,
    dataset_por_lotes,
    entrenar,
    entrenar_fijo,
    importancia_permutacion,
    matriz,
    predecir_por_lotes,
)

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


def test_el_modelo_oficial_usa_15_features_sin_el_tiempo_de_ayer():
    assert len(FEATURES_MODELO) == 15
    assert not any(f.endswith("_ayer") for f in FEATURES_MODELO)
    assert {f"delta_libres_{k}" for k in (15, 30, 60)} <= set(FEATURES_MODELO)
    # Mismo orden que el Parquet, y los grupos del modelo cubren justo sus features.
    assert FEATURES_MODELO == [f for f in FEATURES if f in FEATURES_MODELO]
    assert sorted(c for columnas in GRUPOS_MODELO.values() for c in columnas) == sorted(FEATURES_MODELO)


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


def test_entrenar_fijo_hace_justo_los_arboles_pedidos_sin_mirar_validacion():
    X, y = datos(20_000, 0)
    parametros = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 100,
                  "num_threads": 2, "seed": 0, "verbose": -1}
    modelo = entrenar_fijo(dataset(X, y, COLUMNAS), arboles=37, parametros=parametros)

    assert modelo.num_trees() == 37
    assert modelo.best_iteration == 0  # sin parada temprana: no hay «mejor» iteración
    # Mismos datos y semilla, mismo modelo: el reentrenamiento final es reproducible.
    otro = entrenar_fijo(dataset(X, y, COLUMNAS), arboles=37, parametros=parametros)
    assert np.array_equal(modelo.predict(X), otro.predict(X))
    p = modelo.predict(X)
    assert p[X[:, 0] == 1].mean() == pytest.approx(0.4, abs=0.03)


def parquet_por_bloques(tmp_path, n: int = 6_000, filas_por_bloque: int = 1_000):
    """Parquet con varios bloques: slot_utc creciente, tres columnas y la etiqueta."""
    X, y = datos(n, 7)
    df = pd.DataFrame({"slot_utc": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
                       "senal": X[:, 0], "ruido": X[:, 1], "estacion": X[:, 2].astype("int16"), "llena_60": y})
    ruta = tmp_path / "f.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ruta, row_group_size=filas_por_bloque)
    return ruta, df


def test_secuencia_lee_las_filas_filtradas_en_orden(tmp_path):
    ruta, df = parquet_por_bloques(tmp_path)
    desde, hasta = df["slot_utc"].iloc[1_500], df["slot_utc"].iloc[4_200]
    secuencia = SecuenciaParquet(ruta, COLUMNAS, [("slot_utc", ">=", desde), ("slot_utc", "<", hasta)], batch_size=700)
    esperado = df.iloc[1_500:4_200]

    assert len(secuencia) == 2_700
    X = esperado[COLUMNAS].to_numpy(dtype="float32")
    assert np.array_equal(secuencia[0:2_700], X)                 # tramo que cruza varios bloques
    assert np.array_equal(secuencia[1_234], X[1_234])            # una fila suelta
    assert np.array_equal(secuencia[490:510], X[490:510])        # justo en el cambio de bloque
    assert np.array_equal(np.concatenate(list(secuencia.bloques())), X)
    assert secuencia.leer(["llena_60"])["llena_60"].tolist() == esperado["llena_60"].tolist()


def test_dataset_por_lotes_da_el_mismo_modelo_que_la_matriz(tmp_path):
    ruta, df = parquet_por_bloques(tmp_path)
    corte = df["slot_utc"].iloc[4_000]
    parametros = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 50,
                  "num_threads": 2, "seed": 0, "verbose": -1}
    entrenamiento = df[df["slot_utc"] < corte]
    validacion = df[df["slot_utc"] >= corte]

    # Con la matriz entera en memoria.
    d = dataset(entrenamiento[COLUMNAS].to_numpy("float32"), entrenamiento["llena_60"].to_numpy(), COLUMNAS)
    v = dataset(validacion[COLUMNAS].to_numpy("float32"), validacion["llena_60"].to_numpy(), COLUMNAS, referencia=d)
    con_matriz = entrenar(d, v, parametros, max_arboles=60, paciencia=10, registro=0)

    # Por lotes.
    s_ent = SecuenciaParquet(ruta, COLUMNAS, [("slot_utc", "<", corte)], batch_size=333)
    s_val = SecuenciaParquet(ruta, COLUMNAS, [("slot_utc", ">=", corte)], batch_size=333)
    d2 = dataset_por_lotes(s_ent, s_ent.leer(["llena_60"])["llena_60"].to_numpy())
    v2 = dataset_por_lotes(s_val, s_val.leer(["llena_60"])["llena_60"].to_numpy(), referencia=d2)
    por_lotes = entrenar(d2, v2, parametros, max_arboles=60, paciencia=10, registro=0)

    assert por_lotes.best_iteration == con_matriz.best_iteration
    esperado = con_matriz.predict(validacion[COLUMNAS].to_numpy("float32"), num_iteration=con_matriz.best_iteration)
    obtenido = predecir_por_lotes(por_lotes, s_val, num_iteration=por_lotes.best_iteration)
    assert len(obtenido) == len(validacion)
    assert np.allclose(obtenido, esperado, rtol=0, atol=1e-12)
