"""Fábrica de fotos y zips diminutos con datos inventados (sin datos reales)."""

import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from valenbisi.features_v2 import FEATURES_RETARDOS, FEATURES_V2

CABECERA_ANTIGUA = (
    "gid;name;number_;address;open;available;free;total;ticket;updated_at;"
    "globalid;created_user;created_date;last_edited_user;last_edited_date;geo_shape;geo_point_2d"
)
CABECERA_NUEVA = (
    "Direccion;Numero;Activo;Bicis_disponibles;Espacios_libres;Espacios_totales;"
    "ticket;fecha_actualizacion;geo_shape;geo_point_2d"
)
GEO_SHAPE = '"{""coordinates"": [-0.37, 39.47], ""type"": ""Point""}"'


def csv_antiguo(filas: list[tuple]) -> str:
    """filas: (estacion, activa, bicis, libres, total, actualizada 'DD/MM/YYYY HH:MM:SS')."""
    lineas = [CABECERA_ANTIGUA]
    for est, act, bicis, libres, total, upd in filas:
        lineas.append(
            f"1;{est}_NOMBRE;{est};Calle {est};{act};{bicis};{libres};{total};T;{upd};"
            f"{{X}};;;;;{GEO_SHAPE};39.47,-0.37"
        )
    return "\n".join(lineas) + "\n"


def csv_nuevo(filas: list[tuple]) -> str:
    """Mismo formato de filas que :func:`csv_antiguo`, con el esquema posterior al 2023-05-22."""
    lineas = [CABECERA_NUEVA]
    for est, act, bicis, libres, total, upd in filas:
        lineas.append(
            f"Calle {est};{est};{act};{bicis};{libres};{total};F;{upd};{GEO_SHAPE};39.47,-0.37"
        )
    return "\n".join(lineas) + "\n"


def nombre_foto(fecha: str, hora: str) -> str:
    """fecha 'DD-MM-YYYY', hora 'HH-MM-SS'."""
    return f"valenbici_{fecha}_{hora}.csv"


def escribir_zip(ruta: Path, fotos: list[tuple[str, str]]) -> Path:
    """Escribe un zip con (nombre, contenido) en el orden dado; admite nombres repetidos."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "Duplicate name"
        with zipfile.ZipFile(ruta, "w") as zf:
            for nombre, contenido in fotos:
                zf.writestr(nombre, contenido)
    return ruta


def parquet_features_v2(ruta: Path) -> None:
    """Parquet de features v2 inventado: dic-2024 a may-2025 (entrenamiento hasta feb, validación
    desde mar), 4 slots al día y 4 estaciones, con un 10 % de retardos NaN."""
    rng = np.random.default_rng(0)
    slots = pd.date_range("2024-12-01", "2025-05-31 23:00", freq="6h", tz="Europe/Madrid").tz_convert("UTC")
    df = pd.DataFrame([(s, e) for s in slots for e in range(1, 5)], columns=["slot_utc", "estacion"])
    n = len(df)
    for c in FEATURES_V2:
        df[c] = rng.random(n).astype("float32")
    df["estacion"] = df["estacion"].astype("int16")
    df["libres"] = rng.integers(0, 10, n).astype("int16")
    df["llena"] = df["libres"] == 0
    hueco = rng.random(n) < 0.1
    df.loc[hueco, FEATURES_RETARDOS] = np.nan
    df["llena_60"] = rng.random(n) < np.where(df["libres"] == 0, 0.6, 0.15) + 0.2 * df["vec_ocupacion"]
    df["slot_utc"] = df["slot_utc"].astype("datetime64[us, UTC]")
    pq.write_table(pa.Table.from_pandas(df[["slot_utc", *FEATURES_V2, "llena_60"]], preserve_index=False), ruta,
                   row_group_size=700)
