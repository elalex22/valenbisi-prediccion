"""Estado de las estaciones vecinas en el instante t.

Cada estación tiene fijas sus k vecinas más cercanas (distancia haversine, sin
contarse a sí misma). Para cada fila (estación, t) se miran las fotos de esas
vecinas en el mismo slot t, y en t−60 para la tendencia: nunca posteriores a t.

El estado de todas las estaciones se guarda en una rejilla densa slots ×
estaciones (int16, −1 donde falta la foto), de unos 110 MB para todo el
periodo, y cada lote de filas recoge sus vecinas con indexado de numpy.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from valenbisi.features import EPOCA, INTERVALO, MAX_ESTACION

K_VECINAS = 5
RADIO_TIERRA_KM = 6371.0
FALTA = -1
FEATURES_VECINAS = ["vec_ocupacion", "vec_frac_llenas", "vec_libres_min", "vec_delta_libres_60"]
INTERVALOS_60 = 4


def distancias_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia haversine en km entre todos los pares (matriz len(lat1) × len(lat2))."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype="float64")) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1)[:, None] * np.cos(lat2)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(a))


def vecinas_mas_cercanas(estaciones: pd.DataFrame, k: int = K_VECINAS,
                         candidatas: pd.DataFrame | None = None) -> np.ndarray:
    """Tabla ``MAX_ESTACION × k``: en la fila de cada estación, sus k vecinas más cercanas (−1 si no hay).

    ``candidatas`` son las estaciones que pueden ser vecinas (por defecto, las
    mismas). En la prueba v2, una estación nueva busca vecinas entre las
    conocidas, pero no pasa a ser vecina de ninguna conocida.
    """
    candidatas = estaciones if candidatas is None else candidatas
    if estaciones["estacion"].max() >= MAX_ESTACION or candidatas["estacion"].max() >= MAX_ESTACION:
        raise ValueError(f"Número de estación mayor que {MAX_ESTACION}")
    d = distancias_km(estaciones["lat"], estaciones["lon"], candidatas["lat"], candidatas["lon"])
    ids = candidatas["estacion"].to_numpy()
    d[estaciones["estacion"].to_numpy()[:, None] == ids[None, :]] = np.inf  # nunca vecina de sí misma
    orden = np.argsort(d, axis=1, kind="stable")[:, :k]   # menos de k columnas si hay pocas candidatas
    tabla = np.full((MAX_ESTACION, k), FALTA, dtype="int32")
    cercanas = np.where(np.isfinite(np.take_along_axis(d, orden, axis=1)), ids[orden], FALTA)
    tabla[estaciones["estacion"].to_numpy()[:, None], np.arange(orden.shape[1])] = cercanas
    return tabla


@dataclass
class Rejilla:
    """Estado de todas las estaciones en cada slot: ``libres`` y ``bicis`` (int16, −1 si falta la foto)."""
    inicio: int             # intervalo de 15 min (desde EPOCA) de la primera fila
    columna: np.ndarray     # MAX_ESTACION -> columna de la estación (−1 si no está)
    libres: np.ndarray      # slots × estaciones
    bicis: np.ndarray


def intervalos(slot_utc: pd.Series) -> np.ndarray:
    return ((slot_utc - EPOCA) // INTERVALO).to_numpy(dtype="int64")


def preparar_rejilla(df: pd.DataFrame) -> Rejilla:
    """Rejilla densa con las fotos de ``df`` (``slot_utc``, ``estacion``, ``libres``, ``bicis``).

    ``df`` es el dataset limpio entero: una foto sin t+60 sigue siendo un pasado válido.
    """
    iv = intervalos(df["slot_utc"])
    inicio = int(iv.min())
    estaciones = np.unique(df["estacion"].to_numpy())
    columna = np.full(MAX_ESTACION, FALTA, dtype="int32")
    columna[estaciones] = np.arange(len(estaciones))
    fila, col = iv - inicio, columna[df["estacion"].to_numpy()]
    if pd.Series(fila * len(estaciones) + col).duplicated().any():
        raise ValueError("La rejilla tiene dos fotos en el mismo (slot_utc, estacion)")
    forma = (int(fila.max()) + 1, len(estaciones))
    libres = np.full(forma, FALTA, dtype="int16")
    bicis = np.full(forma, FALTA, dtype="int16")
    libres[fila, col] = df["libres"].to_numpy()
    bicis[fila, col] = df["bicis"].to_numpy()
    return Rejilla(inicio=inicio, columna=columna, libres=libres, bicis=bicis)


def _recoger(rejilla: Rejilla, fila: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(libres, bicis) en float32 con NaN donde falta la foto o la fila/columna cae fuera."""
    dentro = (fila >= 0) & (fila < rejilla.libres.shape[0]) & (col >= 0)
    f, c = np.where(dentro, fila, 0), np.where(dentro, col, 0)
    libres = np.where(dentro, rejilla.libres[f, c], FALTA).astype("float32")
    bicis = np.where(dentro, rejilla.bicis[f, c], FALTA).astype("float32")
    falta = libres == FALTA
    libres[falta], bicis[falta] = np.nan, np.nan
    return libres, bicis


def _media(valores: np.ndarray) -> np.ndarray:
    """Media por fila ignorando NaN; NaN si no queda ninguno (sin avisos de numpy)."""
    validos = ~np.isnan(valores)
    n = validos.sum(axis=1)
    suma = np.where(validos, valores, 0).sum(axis=1)
    return np.where(n > 0, suma / np.maximum(n, 1), np.nan).astype("float32")


def features_vecinas(filas: pd.DataFrame, rejilla: Rejilla, vecinas: np.ndarray) -> pd.DataFrame:
    """Estado de las k vecinas de cada fila en su slot t (y en t−60 para la tendencia).

    Se ignoran las vecinas sin foto; si no queda ninguna, NaN.
    """
    fila = (intervalos(filas["slot_utc"]) - rejilla.inicio)[:, None]
    ids = vecinas[filas["estacion"].to_numpy()]
    col = np.where(ids >= 0, rejilla.columna[np.maximum(ids, 0)], FALTA)
    libres, bicis = _recoger(rejilla, fila, col)
    libres_60, _ = _recoger(rejilla, fila - INTERVALOS_60, col)

    funcionan = bicis + libres
    ocupacion = np.where(funcionan > 0, bicis / np.where(funcionan > 0, funcionan, 1), np.nan)
    llenas = np.where(np.isnan(libres), np.nan, (libres == 0).astype("float32"))
    minimo = np.where(np.isnan(libres), np.inf, libres).min(axis=1)
    return pd.DataFrame({
        "vec_ocupacion": _media(ocupacion.astype("float32")),
        "vec_frac_llenas": _media(llenas),
        "vec_libres_min": np.where(np.isinf(minimo), np.nan, minimo).astype("float32"),
        "vec_delta_libres_60": _media(libres - libres_60),
    }, index=filas.index)
