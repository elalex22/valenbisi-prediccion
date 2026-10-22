"""Reglas de limpieza aprobadas (ver CLAUDE.md), una función por regla.

Cada regla recibe un DataFrame con las columnas de :mod:`valenbisi.lectura`
y devuelve las filas que se conservan. :func:`aplicar_reglas` las encadena y
anota cuántas filas había antes y después de cada una.

Las reglas que miran una fila sola se pueden aplicar zip a zip (``REGLAS_POR_TROZO``).
Las que comparan fotos entre sí necesitan todo el periodo junto (``REGLAS_GLOBALES``).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

ESTACION_PRUEBAS = 299
MAX_ANTIGUEDAD = pd.Timedelta(hours=24)

Regla = Callable[[pd.DataFrame], pd.DataFrame]


def quitar_hora_repetida(df: pd.DataFrame) -> pd.DataFrame:
    """Fotos de la hora repetida de octubre, que quedaron sin ``ts_utc``."""
    return df[df["ts_utc"].notna()]


def quitar_estacion_pruebas(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["estacion"] != ESTACION_PRUEBAS]


def quitar_inactivas(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["activa"]]


def quitar_desactualizadas(df: pd.DataFrame) -> pd.DataFrame:
    """Registros cuya última actualización tiene más de 24 h (o no se pudo leer).

    Se compara en hora local: el desfase de 1 h del cambio de hora es irrelevante frente a 24 h.
    """
    antiguedad = df["ts"] - df["fecha_actualizacion"]
    return df[antiguedad.notna() & (antiguedad <= MAX_ANTIGUEDAD)]


def quitar_fotos_identicas(df: pd.DataFrame) -> pd.DataFrame:
    """Fotos iguales a la anterior (en orden de ``ts_utc``): el feed estaba congelado.

    Dos fotos son iguales si tienen el mismo conjunto de
    (estacion, bicis, libres, fecha_actualizacion). Incluir la fecha de
    actualización evita borrar una foto real en la que no se movió ninguna bici.
    """
    hash_fila = pd.util.hash_pandas_object(
        df[["estacion", "bicis", "libres", "fecha_actualizacion"]], index=False
    ).to_numpy()
    # Firma de cada foto independiente del orden de las filas: suma de hashes y número de filas.
    firma = (
        pd.DataFrame({"ts_utc": df["ts_utc"].to_numpy(), "hash": hash_fila})
        .groupby("ts_utc", sort=True)["hash"]
        .agg(["sum", "size"])
    )
    suma = firma["sum"].to_numpy()
    filas = firma["size"].to_numpy()
    igual_a_anterior = np.zeros(len(firma), dtype=bool)
    igual_a_anterior[1:] = (suma[1:] == suma[:-1]) & (filas[1:] == filas[:-1])
    return df[~df["ts_utc"].isin(firma.index[igual_a_anterior])]


def quitar_sin_bicis_ni_huecos(df: pd.DataFrame) -> pd.DataFrame:
    """0 bicis y 0 anclajes libres: anclajes averiados, no es una estación llena."""
    return df[~(df["bicis"].eq(0) & df["libres"].eq(0))]


def quitar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Comprobación defensiva: una sola fila por (ts_utc, estacion)."""
    return df.drop_duplicates(["ts_utc", "estacion"])


REGLAS_POR_TROZO: list[tuple[str, Regla]] = [
    ("Hora repetida de octubre", quitar_hora_repetida),
    ("Estación de pruebas (299)", quitar_estacion_pruebas),
    ("Registros inactivos", quitar_inactivas),
    ("Actualización con más de 24 h", quitar_desactualizadas),
]
REGLAS_GLOBALES: list[tuple[str, Regla]] = [
    ("Foto idéntica a la anterior", quitar_fotos_identicas),
    ("0 bicis y 0 huecos libres", quitar_sin_bicis_ni_huecos),
    ("Duplicados (ts_utc, estación)", quitar_duplicados),
]


def aplicar_reglas(
    df: pd.DataFrame,
    reglas: list[tuple[str, Regla]],
    conteo: dict[str, list[int]],
) -> pd.DataFrame:
    """Aplica las reglas en orden y suma en ``conteo[nombre] = [antes, despues]``.

    ``conteo`` se acumula entre llamadas, así que sirve para procesar zip a zip.
    """
    for nombre, regla in reglas:
        antes = len(df)
        df = regla(df)
        acumulado = conteo.setdefault(nombre, [0, 0])
        acumulado[0] += antes
        acumulado[1] += len(df)
    return df


def marcar_llena(df: pd.DataFrame) -> pd.DataFrame:
    """Llena = estación activa con 0 anclajes libres (tras la limpieza todas están activas)."""
    return df.assign(llena=df["libres"].eq(0))
