"""Features candidatas de la v2: las 20 de la v1 más meteorología horaria, vecinas y retardos largos.

Todas usan solo datos con ``slot_utc ≤ t`` (ver cada módulo). La ablación en
validación decide cuáles se quedan.
"""

import numpy as np
import pandas as pd

from valenbisi.calendario import features_calendario
from valenbisi.features import FEATURES, Referencia, construir_features, deltas_libres, estado_actual, retardos_largos
from valenbisi.open_meteo import FEATURES_METEO, tiempo_hora_pasada
from valenbisi.vecinas import FEATURES_VECINAS, Rejilla, features_vecinas

FEATURES_RETARDOS = ["libres_obj_24h", "cambio_24h", "libres_obj_7d", "cambio_7d"]
FEATURES_V2 = [*FEATURES, *FEATURES_METEO, *FEATURES_VECINAS, *FEATURES_RETARDOS]

# Las 23 que eligió la ablación v2 (scripts/ablacion_v2.py, CLAUDE.md): las 15 de la v1
# (sin el tiempo de ayer), las vecinas y los retardos. La meteorología horaria queda fuera.
FEATURES_V2_FINALES = [
    "llena", "libres", "bicis", "total", "ocupacion",
    "delta_libres_15", "delta_libres_30", "delta_libres_60",
    "estacion", "lat", "lon",
    "minuto_del_dia", "dia_semana", "mes", "festivo",
    *FEATURES_VECINAS, *FEATURES_RETARDOS,
]


def construir_features_v2(
    filas: pd.DataFrame, referencia: Referencia, rejilla: Rejilla, vecinas: np.ndarray,
    tiempo: pd.DataFrame, horario: pd.DataFrame, estaciones: pd.DataFrame,
) -> pd.DataFrame:
    """``slot_utc``, ``estacion``, las ``FEATURES_V2`` y ``llena_60`` si ``filas`` la trae."""
    v1 = construir_features(filas, referencia, tiempo, estaciones)
    nuevas = [
        tiempo_hora_pasada(filas["slot_utc"], horario),
        features_vecinas(filas, rejilla, vecinas),
        retardos_largos(filas, referencia),
    ]
    df = pd.concat([v1, *nuevas], axis=1)
    return df[["slot_utc", *FEATURES_V2, *(["llena_60"] if "llena_60" in filas else [])]]


def features_finales_v2(
    filas: pd.DataFrame, referencia: Referencia, rejilla: Rejilla, vecinas: np.ndarray, estaciones: pd.DataFrame,
) -> pd.DataFrame:
    """Como :func:`construir_features_v2`, pero solo con ``FEATURES_V2_FINALES``.

    Las 23 no usan meteorología, así que predecir no depende de AEMET ni de Open-Meteo.
    """
    ubicacion = estaciones.set_index("estacion")[["lat", "lon"]].astype("float32")
    partes = [
        estado_actual(filas),
        deltas_libres(filas, referencia),
        ubicacion.reindex(filas["estacion"].to_numpy()).set_axis(filas.index),
        features_calendario(filas["slot_utc"]),
        features_vecinas(filas, rejilla, vecinas),
        retardos_largos(filas, referencia),
    ]
    etiqueta = ["llena_60"] if "llena_60" in filas else []
    df = pd.concat([filas[["slot_utc", "estacion", *etiqueta]], *partes], axis=1)
    return df[["slot_utc", *FEATURES_V2_FINALES, *etiqueta]]
