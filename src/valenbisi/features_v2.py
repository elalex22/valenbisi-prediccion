"""Features candidatas de la v2: las 20 de la v1 más meteorología horaria, vecinas y retardos largos.

Todas usan solo datos con ``slot_utc ≤ t`` (ver cada módulo). La ablación en
validación decide cuáles se quedan.
"""

import numpy as np
import pandas as pd

from valenbisi.features import FEATURES, Referencia, construir_features, retardos_largos
from valenbisi.open_meteo import FEATURES_METEO, tiempo_hora_pasada
from valenbisi.vecinas import FEATURES_VECINAS, Rejilla, features_vecinas

FEATURES_RETARDOS = ["libres_obj_24h", "cambio_24h", "libres_obj_7d", "cambio_7d"]
FEATURES_V2 = [*FEATURES, *FEATURES_METEO, *FEATURES_VECINAS, *FEATURES_RETARDOS]


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
