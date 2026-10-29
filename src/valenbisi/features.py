"""Features de cada fila (estación, t) para el modelo, usando solo datos con ``slot_utc ≤ t``.

- Estado actual: la fila de t (``llena``, ``libres``, ``bicis``, ``total``, ``ocupacion``).
- Tendencia: ``libres(t) − libres(t−k)`` para k = 15, 30 y 60 min, buscando la foto
  de t−k de la misma estación. Si no existe, queda NaN: nunca se rellena con otra
  foto ni con 0, que significaría «no ha cambiado».
- Estación: el identificador (categórico) y su ubicación.
- Calendario: ver :mod:`valenbisi.calendario`.
- Meteorología: la del día anterior en hora local. La del mismo día incluye
  horas que aún no han pasado (la lluvia de la tarde, la máxima), sería una fuga.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from valenbisi.calendario import features_calendario
from valenbisi.evaluacion import ZONA
from valenbisi.tiempo import VARIABLES

RETARDOS_MIN = (15, 30, 60)
INTERVALO = pd.Timedelta(minutes=15)
INTERVALOS_DIA = 96
INTERVALOS_HORIZONTE = 4  # los 60 min del objetivo
EPOCA = pd.Timestamp("2000-01-01", tz="UTC")
# Las claves (intervalo, estación) se codifican como intervalo * MAX_ESTACION + estación.
MAX_ESTACION = 1000

FEATURES = [
    "llena", "libres", "bicis", "total", "ocupacion",
    *(f"delta_libres_{k}" for k in RETARDOS_MIN),
    "estacion", "lat", "lon",
    "minuto_del_dia", "dia_semana", "mes", "festivo",
    *(f"{v}_ayer" for v in VARIABLES),
]
CATEGORICAS = ["estacion"]


def claves(slot_utc: pd.Series, estacion: pd.Series) -> np.ndarray:
    """Un entero por (intervalo de 15 min, estación); dos filas de la misma estación separadas
    k intervalos difieren en ``k * MAX_ESTACION``."""
    if estacion.max() >= MAX_ESTACION:
        raise ValueError(f"Número de estación mayor que {MAX_ESTACION}: {estacion.max()}")
    intervalo = (slot_utc - EPOCA) // INTERVALO
    return intervalo.to_numpy(dtype="int64") * MAX_ESTACION + estacion.to_numpy(dtype="int64")


@dataclass
class Referencia:
    """Fotos donde se busca el pasado, ordenadas por clave para buscar con ``searchsorted``."""
    claves: np.ndarray   # int64, ordenadas y sin repetidos
    libres: np.ndarray   # libres de cada clave


def preparar_referencia(df: pd.DataFrame) -> Referencia:
    """Ordena una vez las fotos de ``df`` (con ``slot_utc``, ``estacion`` y ``libres``).

    ``df`` es el dataset limpio entero, no solo las filas con etiqueta: una foto
    sin t+60 sigue siendo un pasado válido.
    """
    clave = claves(df["slot_utc"], df["estacion"])
    orden = np.argsort(clave, kind="stable")
    clave = clave[orden]
    if (np.diff(clave) == 0).any():
        raise ValueError("La referencia tiene dos fotos en el mismo (slot_utc, estacion)")
    return Referencia(claves=clave, libres=df["libres"].to_numpy()[orden])


def libres_en(filas: pd.DataFrame, referencia: Referencia, desplazamiento: int) -> np.ndarray:
    """``libres`` de la misma estación ``desplazamiento`` intervalos de 15 min antes de cada fila.

    Se busca por clave exacta con ``searchsorted``, sin un merge que duplique la
    tabla. Si falta esa foto, NaN: nunca se rellena con otra.
    """
    buscada = claves(filas["slot_utc"], filas["estacion"]) - desplazamiento * MAX_ESTACION
    pos = np.searchsorted(referencia.claves, buscada).clip(max=len(referencia.claves) - 1)
    existe = referencia.claves[pos] == buscada
    return np.where(existe, referencia.libres[pos], np.nan).astype("float32")


def deltas_libres(filas: pd.DataFrame, referencia: Referencia) -> pd.DataFrame:
    """``delta_libres_k`` de cada fila de ``filas``, con las fotos anteriores de ``referencia``."""
    libres = filas["libres"].to_numpy(dtype="float32")
    salida = {f"delta_libres_{k}": libres - libres_en(filas, referencia, k // 15) for k in RETARDOS_MIN}
    return pd.DataFrame(salida, index=filas.index)


def retardos_largos(filas: pd.DataFrame, referencia: Referencia) -> pd.DataFrame:
    """La misma estación ayer y hace una semana, **a la hora del objetivo** (t + 60 min).

    Lo que mejor anticipa si estará llena a t+60 es cómo estaba a esa misma hora
    otros días, y sigue siendo pasado: t + 60 min − 24 h = t − 23 h.

    - ``libres_obj_24h`` / ``libres_obj_7d``: libres en t+60−24 h / t+60−7 d.
    - ``cambio_24h`` / ``cambio_7d``: cuánto cambió libres en esa hora aquel día,
      de t−24 h a t+60−24 h (y lo mismo hace 7 días).

    En UTC: los dos días de cambio de hora, «ayer a la misma hora» se desplaza una
    hora local.
    """
    salida = {}
    for nombre, dias in (("24h", 1), ("7d", 7)):
        atras = dias * INTERVALOS_DIA
        en_objetivo = libres_en(filas, referencia, atras - INTERVALOS_HORIZONTE)
        salida[f"libres_obj_{nombre}"] = en_objetivo
        salida[f"cambio_{nombre}"] = en_objetivo - libres_en(filas, referencia, atras)
    return pd.DataFrame(salida, index=filas.index)


def estado_actual(filas: pd.DataFrame) -> pd.DataFrame:
    """La foto de t. ``ocupacion`` se divide entre los anclajes que funcionan
    (``bicis + libres``), así que una estación sin huecos libres da 1,0 aunque
    tenga anclajes averiados."""
    funcionan = (filas["bicis"] + filas["libres"]).astype("float32")
    return pd.DataFrame({
        "llena": filas["llena"],
        "libres": filas["libres"],
        "bicis": filas["bicis"],
        "total": filas["total"],
        "ocupacion": (filas["bicis"] / funcionan.where(funcionan > 0)).astype("float32"),
    }, index=filas.index)


def tiempo_de_ayer(slot_utc: pd.Series, tiempo: pd.DataFrame) -> pd.DataFrame:
    """Variables de AEMET del día local anterior a t (NaN si ese día no está)."""
    ayer = slot_utc.dt.tz_convert(ZONA).dt.tz_localize(None).dt.normalize() - pd.Timedelta(days=1)
    por_dia = tiempo.set_index("fecha")[VARIABLES].add_suffix("_ayer")
    return por_dia.reindex(ayer.to_numpy()).set_axis(slot_utc.index)


def construir_features(
    filas: pd.DataFrame, referencia: Referencia, tiempo: pd.DataFrame, estaciones: pd.DataFrame
) -> pd.DataFrame:
    """``slot_utc``, ``estacion``, las ``FEATURES`` y ``llena_60`` si ``filas`` la trae."""
    ubicacion = estaciones.set_index("estacion")[["lat", "lon"]].astype("float32")
    partes = [
        estado_actual(filas),
        deltas_libres(filas, referencia),
        ubicacion.reindex(filas["estacion"].to_numpy()).set_axis(filas.index),
        features_calendario(filas["slot_utc"]),
        tiempo_de_ayer(filas["slot_utc"], tiempo),
    ]
    df = pd.concat([filas[["slot_utc", "estacion"]], *partes], axis=1)
    if "llena_60" in filas:
        df["llena_60"] = filas["llena_60"]
    return df[["slot_utc", *FEATURES, *(["llena_60"] if "llena_60" in filas else [])]]
