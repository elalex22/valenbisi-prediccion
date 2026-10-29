"""Primer modelo: LightGBM sobre las features de :mod:`valenbisi.features`.

Desbalance (~6 % de positivos): se entrena con la pérdida logística **sin pesos**.
Su mínimo está en la probabilidad real, así que las predicciones salen
calibradas. ``is_unbalance``, ``scale_pos_weight`` o quitar negativos inflarían
las probabilidades (y la AUC-PR, que solo depende del orden, apenas mejoraría).
"""

from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pyarrow as pa
from sklearn.metrics import average_precision_score

from valenbisi.features import CATEGORICAS, FEATURES, RETARDOS_MIN
from valenbisi.memoria import comprobar_memoria
from valenbisi.tiempo import VARIABLES

PARAMETROS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    # Con millones de filas, hojas de al menos 1000 evitan memorizar casos sueltos.
    "min_data_in_leaf": 1000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "num_threads": 12,
    "seed": 42,
    "verbose": -1,
}
MAX_ARBOLES = 3000
PACIENCIA = 100

# Features que se solapan: al barajar una sola, las demás del grupo la suplen.
GRUPOS = {
    "estado actual": ["llena", "libres", "bicis", "total", "ocupacion"],
    "tendencia": [f"delta_libres_{k}" for k in RETARDOS_MIN],
    "estación y ubicación": ["estacion", "lat", "lon"],
    "calendario": ["minuto_del_dia", "dia_semana", "mes", "festivo"],
    "tiempo de ayer": [f"{v}_ayer" for v in VARIABLES],
}

# Grupos que se quedan fuera del modelo oficial, aunque estén en el Parquet de
# features. El tiempo de ayer no aporta y empeora validación (scripts/ablacion_lightgbm.py):
# los árboles lo usaban para reconocer días concretos de entrenamiento.
GRUPOS_DESCARTADOS = ("tiempo de ayer",)
GRUPOS_MODELO = {g: columnas for g, columnas in GRUPOS.items() if g not in GRUPOS_DESCARTADOS}
FEATURES_MODELO = [f for f in FEATURES if any(f in columnas for columnas in GRUPOS_MODELO.values())]


def matriz(tabla: pa.Table, columnas: list[str] = FEATURES_MODELO) -> np.ndarray:
    """Columnas de ``tabla`` como una matriz float32 por filas (lo que LightGBM lee sin copiar).

    Se rellena columna a columna para no tener a la vez un DataFrame entero y la
    matriz. Los nulos de Arrow pasan a NaN y los booleanos a 0/1.
    """
    X = np.empty((tabla.num_rows, len(columnas)), dtype=np.float32)
    for j, columna in enumerate(columnas):
        X[:, j] = tabla.column(columna).to_numpy()
    return X


def dataset(X: np.ndarray, y: np.ndarray, columnas: list[str] = FEATURES_MODELO,
            referencia: lgb.Dataset | None = None) -> lgb.Dataset:
    """Dataset de LightGBM ya construido (los histogramas), para poder soltar ``X`` justo después.

    ``referencia`` hace que validación use los mismos cortes que entrenamiento.
    """
    datos = lgb.Dataset(
        X, label=y, feature_name=columnas, categorical_feature=[c for c in CATEGORICAS if c in columnas],
        reference=referencia, free_raw_data=True, params={"max_bin": PARAMETROS["max_bin"], "verbose": -1},
    )
    return datos.construct()


def vigilar_memoria(minimo_libre_gb: float, cada: int = 25) -> Callable:
    """Callback que para el entrenamiento si el equipo se queda sin RAM."""
    def callback(entorno) -> None:
        if entorno.iteration % cada == 0:
            comprobar_memoria(minimo_libre_gb)
    return callback


def entrenar(entrenamiento: lgb.Dataset, validacion: lgb.Dataset, parametros: dict = PARAMETROS,
             max_arboles: int = MAX_ARBOLES, paciencia: int = PACIENCIA, callbacks: list | None = None,
             registro: int = 100) -> lgb.Booster:
    """Entrena con parada temprana: para cuando la logloss de validación lleva
    ``paciencia`` árboles sin mejorar y se queda con el mejor número de árboles."""
    return lgb.train(
        parametros, entrenamiento, num_boost_round=max_arboles,
        valid_sets=[validacion], valid_names=["validacion"],
        callbacks=[lgb.early_stopping(paciencia, verbose=False), lgb.log_evaluation(registro), *(callbacks or [])],
    )


def importancia_permutacion(
    predecir: Callable[[np.ndarray], np.ndarray], X: np.ndarray, y: np.ndarray,
    grupos: dict[str, list[int]], semilla: int = 0,
) -> tuple[float, dict[str, float]]:
    """AUC-PR sin tocar nada y cuánto cae al barajar las columnas de cada grupo.

    Las columnas de un grupo se barajan con el mismo orden de filas, así que siguen
    siendo coherentes entre sí pero dejan de corresponder a su fila.
    """
    rng = np.random.default_rng(semilla)
    base = average_precision_score(y, predecir(X))
    caidas = {}
    for nombre, columnas in grupos.items():
        barajada = X.copy()
        barajada[:, columnas] = X[np.ix_(rng.permutation(len(X)), columnas)]
        caidas[nombre] = base - average_precision_score(y, predecir(barajada))
    return base, caidas
