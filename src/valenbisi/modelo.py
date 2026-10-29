"""Primer modelo: LightGBM sobre las features de :mod:`valenbisi.features`.

Desbalance (~6 % de positivos): se entrena con la pérdida logística **sin pesos**.
Su mínimo está en la probabilidad real, así que las predicciones salen
calibradas. ``is_unbalance``, ``scale_pos_weight`` o quitar negativos inflarían
las probabilidades (y la AUC-PR, que solo depende del orden, apenas mejoraría).
"""

from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
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


_COMPARACIONES = {">=": pc.greater_equal, ">": pc.greater, "<": pc.less, "<=": pc.less_equal, "==": pc.equal}


class SecuenciaParquet(lgb.Sequence):
    """Las filas de un Parquet que cumplen ``filtros``, leídas por bloques (row groups) como matriz float32.

    Sirve para construir el Dataset de LightGBM sin tener la matriz entera en
    memoria: con 35 columnas y 15 M filas serían 2,1 GB. Se guarda solo el último
    bloque leído; LightGBM recorre las filas en orden, así que cada bloque se lee
    una vez por pasada. ``filtros`` son condiciones (columna, operador, valor)
    unidas con «y», como las de :func:`valenbisi.evaluacion.limites`.
    """

    def __init__(self, ruta, columnas: list[str], filtros: list[tuple], batch_size: int = 100_000):
        self.archivo = pq.ParquetFile(ruta)
        self.columnas = list(columnas)
        self.batch_size = batch_size
        usadas = sorted({c for c, _, _ in filtros}) or [self.archivo.schema_arrow.names[0]]
        self.filas_bloque = []
        for g in range(self.archivo.num_row_groups):
            tabla = self.archivo.read_row_group(g, columns=usadas)
            mascara = pa.array(np.ones(tabla.num_rows, dtype=bool))
            for columna, operador, valor in filtros:
                mascara = pc.and_(mascara, _COMPARACIONES[operador](tabla.column(columna), pa.scalar(valor)))
            self.filas_bloque.append(np.flatnonzero(mascara.to_numpy(zero_copy_only=False)))
        self.fin = np.cumsum([len(f) for f in self.filas_bloque])
        self.inicio = self.fin - [len(f) for f in self.filas_bloque]
        self._cache: tuple[int, np.ndarray] | None = None

    def __len__(self) -> int:
        return int(self.fin[-1]) if len(self.fin) else 0

    def _tabla(self, g: int, columnas: list[str]) -> pa.Table:
        return self.archivo.read_row_group(g, columns=columnas).take(self.filas_bloque[g])

    def bloque(self, g: int) -> np.ndarray:
        if self._cache is None or self._cache[0] != g:
            self._cache = (g, matriz(self._tabla(g, self.columnas), self.columnas))
        return self._cache[1]

    def bloques(self):
        """Las matrices de cada bloque, en orden (para predecir sin cargarlo todo)."""
        for g in range(len(self.filas_bloque)):
            if len(self.filas_bloque[g]):
                yield self.bloque(g)

    def leer(self, columnas: list[str]) -> pd.DataFrame:
        """Otras columnas de las mismas filas y en el mismo orden (etiqueta, slot_utc, libres…)."""
        partes = [self._tabla(g, columnas).to_pandas() for g in range(len(self.filas_bloque))]
        return pd.concat(partes, ignore_index=True)

    def __getitem__(self, indice):
        if isinstance(indice, slice):
            inicio, fin, paso = indice.indices(len(self))
            if paso != 1:
                raise ValueError("Solo se admiten tramos consecutivos")
            partes, i = [], inicio
            while i < fin:
                g = int(np.searchsorted(self.fin, i, side="right"))
                hasta = min(fin, int(self.fin[g]))
                partes.append(self.bloque(g)[i - self.inicio[g]:hasta - self.inicio[g]])
                i = hasta
            return np.concatenate(partes) if partes else np.empty((0, len(self.columnas)), dtype=np.float32)
        # Fila suelta: LightGBM la usa para muestrear los cortes y la exige en float64 (conversión exacta).
        indice = int(indice)
        g = int(np.searchsorted(self.fin, indice, side="right"))
        return self.bloque(g)[indice - self.inicio[g]].astype(np.float64)


def dataset_por_lotes(secuencia: SecuenciaParquet, y: np.ndarray,
                      referencia: lgb.Dataset | None = None) -> lgb.Dataset:
    """Como :func:`dataset`, pero leyendo las filas de ``secuencia`` por bloques."""
    datos = lgb.Dataset(
        [secuencia], label=y, feature_name=secuencia.columnas,
        categorical_feature=[c for c in CATEGORICAS if c in secuencia.columnas],
        reference=referencia, params={"max_bin": PARAMETROS["max_bin"], "verbose": -1},
    )
    return datos.construct()


def predecir_por_lotes(modelo: lgb.Booster, secuencia: SecuenciaParquet,
                       num_iteration: int | None = None) -> np.ndarray:
    """Probabilidades de todas las filas de ``secuencia``, bloque a bloque (con ``num_iteration`` árboles)."""
    partes = [modelo.predict(X, num_iteration=num_iteration, num_threads=PARAMETROS["num_threads"])
              for X in secuencia.bloques()]
    return np.concatenate(partes) if partes else np.empty(0)


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


def entrenar_fijo(entrenamiento: lgb.Dataset, arboles: int, parametros: dict = PARAMETROS,
                  callbacks: list | None = None) -> lgb.Booster:
    """Entrena exactamente ``arboles`` árboles, sin conjunto de validación ni parada temprana.

    Para el modelo final: el número de árboles se fija antes con la parada
    temprana en validación, y luego se reentrena con más datos sin volver a mirar
    ningún conjunto de evaluación.
    """
    return lgb.train(parametros, entrenamiento, num_boost_round=arboles, callbacks=callbacks)


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
