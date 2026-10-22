"""Lectura de los zips brutos de Valenbisi y unificación de los dos esquemas.

Cada zip contiene CSV (separador ``;``), uno por foto de todas las estaciones.
La hora de la foto sale del nombre del fichero, en hora local de Europe/Madrid.
"""

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

ZONA = "Europe/Madrid"
FORMATO_FECHA = "%d/%m/%Y %H:%M:%S"
PATRON_NOMBRE = re.compile(r"valenbici_(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})\.csv$")

# Columnas originales -> columnas comunes (el esquema cambió el 2023-05-22).
ESQUEMA_ANTIGUO = {
    "number_": "estacion",
    "address": "direccion",
    "open": "activa",
    "available": "bicis",
    "free": "libres",
    "total": "total",
    "updated_at": "fecha_actualizacion",
    "geo_point_2d": "geo",
}
ESQUEMA_NUEVO = {
    "Numero": "estacion",
    "Direccion": "direccion",
    "Activo": "activa",
    "Bicis_disponibles": "bicis",
    "Espacios_libres": "libres",
    "Espacios_totales": "total",
    "fecha_actualizacion": "fecha_actualizacion",
    "geo_point_2d": "geo",
}

COLUMNAS = [
    "ts", "estacion", "direccion", "activa", "bicis", "libres", "total",
    "fecha_actualizacion", "lat", "lon",
]


def ts_desde_nombre(nombre: str) -> pd.Timestamp:
    """Hora local de la foto a partir de ``valenbici_DD-MM-YYYY_HH-MM-SS.csv``."""
    m = PATRON_NOMBRE.search(nombre)
    if m is None:
        raise ValueError(f"Nombre de foto no reconocido: {nombre}")
    return pd.to_datetime(m.group(1), format="%d-%m-%Y_%H-%M-%S")


def detectar_esquema(cabecera: str) -> dict[str, str]:
    columnas = cabecera.strip().split(";")
    if "number_" in columnas:
        return ESQUEMA_ANTIGUO
    if "Numero" in columnas:
        return ESQUEMA_NUEVO
    raise ValueError(f"Esquema desconocido: {cabecera[:80]}")


def _vacio() -> pd.DataFrame:
    return pd.DataFrame({
        "ts": pd.Series(dtype="datetime64[us]"),
        "estacion": pd.Series(dtype="int16"),
        "direccion": pd.Series(dtype="str"),
        "activa": pd.Series(dtype="bool"),
        "bicis": pd.Series(dtype="int16"),
        "libres": pd.Series(dtype="int16"),
        "total": pd.Series(dtype="int16"),
        "fecha_actualizacion": pd.Series(dtype="datetime64[us]"),
        "lat": pd.Series(dtype="float64"),
        "lon": pd.Series(dtype="float64"),
    })


def leer_csv(contenido: bytes, ts: pd.Timestamp) -> pd.DataFrame:
    """Una foto (CSV de cualquiera de los dos esquemas) con las columnas comunes."""
    texto = contenido.decode("utf-8")
    if not texto.strip():
        return _vacio()
    esquema = detectar_esquema(texto.split("\n", 1)[0])
    df = pd.read_csv(io.StringIO(texto), sep=";", dtype=str, usecols=list(esquema))
    if df.empty:
        return _vacio()
    df = df.rename(columns=esquema)
    geo = df["geo"].str.split(",", expand=True).astype("float64")
    return pd.DataFrame({
        "ts": pd.Series(ts, index=df.index).astype("datetime64[us]"),
        "estacion": df["estacion"].astype("int16"),
        "direccion": df["direccion"],
        "activa": df["activa"].eq("T"),
        "bicis": df["bicis"].astype("int16"),
        "libres": df["libres"].astype("int16"),
        "total": df["total"].astype("int16"),
        "fecha_actualizacion": pd.to_datetime(
            df["fecha_actualizacion"], format=FORMATO_FECHA, errors="coerce"
        ).astype("datetime64[us]"),
        "lat": geo[0],
        "lon": geo[1],
    })[COLUMNAS]


def a_utc(ts: pd.Series) -> tuple[pd.Series, int]:
    """Convierte horas locales de Europe/Madrid a UTC.

    La hora repetida de octubre no se puede resolver con fiabilidad y queda
    como NaT (la limpieza la descarta). Una hora inexistente de marzo se
    desplaza a las 03:00. Devuelve también cuántas horas inexistentes había.
    """
    local = ts.dt.tz_localize(ZONA, ambiguous="NaT", nonexistent="shift_forward")
    sin_desplazar = ts.dt.tz_localize(ZONA, ambiguous="NaT", nonexistent="NaT")
    inexistentes = int((local.notna() & sin_desplazar.isna()).sum())
    return local.dt.tz_convert("UTC"), inexistentes


def leer_zip(ruta: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Todas las fotos de un zip, con ``ts_utc``, y un pequeño recuento.

    Se recorre ``infolist()`` y no los nombres: un zip podría traer dos
    ficheros con el mismo nombre y ``read(nombre)`` solo devolvería el último.
    """
    trozos = []
    info = {"fotos": 0, "fotos_vacias": 0, "fotos_hora_repetida": 0, "fotos_hora_inexistente": 0}
    with zipfile.ZipFile(ruta) as zf:
        for entrada in zf.infolist():
            if not entrada.filename.endswith(".csv"):
                continue
            ts = ts_desde_nombre(entrada.filename)
            with zf.open(entrada) as f:
                foto = leer_csv(f.read(), ts)
            info["fotos"] += 1
            if foto.empty:
                info["fotos_vacias"] += 1
            else:
                trozos.append(foto)
    if not trozos:
        df = _vacio()
        df.insert(1, "ts_utc", pd.Series(dtype="datetime64[us, UTC]"))
        return df, info

    df = pd.concat(trozos, ignore_index=True)
    horas = pd.Series(df["ts"].unique())
    horas_utc, inexistentes = a_utc(horas)
    info["fotos_hora_repetida"] = int(horas_utc.isna().sum())
    info["fotos_hora_inexistente"] = inexistentes
    df.insert(1, "ts_utc", df["ts"].map(dict(zip(horas, horas_utc))).astype("datetime64[us, UTC]"))
    return df, info


def zips_en_orden(carpeta: Path) -> list[Path]:
    """Zips ordenados por la fecha de su nombre (``DD-MM-YYYY.zip``)."""
    return sorted(
        Path(carpeta).glob("*.zip"),
        key=lambda p: pd.to_datetime(p.stem, format="%d-%m-%Y"),
    )
