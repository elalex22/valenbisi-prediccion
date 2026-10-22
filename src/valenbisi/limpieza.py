"""Reglas de limpieza aprobadas (ver CLAUDE.md), una función por regla.

Cada regla recibe un DataFrame con las columnas de :mod:`valenbisi.lectura`
y devuelve las filas que se conservan. :func:`aplicar_reglas` las encadena y
anota cuántas filas había antes y después de cada una.

Todas se aplican zip a zip, en orden cronológico, para no tener todo el
periodo en memoria. La única que necesita contexto (fotos idénticas)
recuerda la última foto del trozo anterior.
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


class QuitarFotosIdenticas:
    """Fotos iguales a la anterior (en orden de ``ts_utc``): el feed estaba congelado.

    Dos fotos son iguales si tienen el mismo conjunto de
    (estacion, bicis, libres, fecha_actualizacion). Incluir la fecha de
    actualización evita borrar una foto real en la que no se movió ninguna bici.

    Recuerda la firma de la última foto vista, así que al procesar zip a zip
    también compara la primera foto de un zip con la última del anterior.
    Por eso hay que llamarla con los trozos en orden cronológico.
    """

    def __init__(self) -> None:
        self._ultima: tuple[int, int] | None = None

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
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
        if self._ultima is not None:
            igual_a_anterior[0] = (suma[0], filas[0]) == self._ultima
        self._ultima = (suma[-1], filas[-1])
        return df[~df["ts_utc"].isin(firma.index[igual_a_anterior])]


def quitar_fotos_identicas(df: pd.DataFrame) -> pd.DataFrame:
    """:class:`QuitarFotosIdenticas` sobre un solo DataFrame, sin memoria previa."""
    return QuitarFotosIdenticas()(df)


def quitar_sin_bicis_ni_huecos(df: pd.DataFrame) -> pd.DataFrame:
    """0 bicis y 0 anclajes libres: anclajes averiados, no es una estación llena."""
    return df[~(df["bicis"].eq(0) & df["libres"].eq(0))]


def quitar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Comprobación defensiva: una sola fila por (ts_utc, estacion).

    Basta mirar dentro de cada trozo porque los zips no se solapan en el
    tiempo (``construir_dataset`` lo comprueba).
    """
    return df.drop_duplicates(["ts_utc", "estacion"])


def crear_reglas() -> list[tuple[str, Regla]]:
    """Las reglas en el orden aprobado. Crear una lista nueva por ejecución,
    porque la de fotos idénticas guarda estado entre trozos."""
    return [
        ("Hora repetida de octubre", quitar_hora_repetida),
        ("Estación de pruebas (299)", quitar_estacion_pruebas),
        ("Registros inactivos", quitar_inactivas),
        ("Actualización con más de 24 h", quitar_desactualizadas),
        ("Foto idéntica a la anterior", QuitarFotosIdenticas()),
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
