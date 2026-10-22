"""Fábrica de fotos y zips diminutos con datos inventados (sin datos reales)."""

import warnings
import zipfile
from pathlib import Path

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
