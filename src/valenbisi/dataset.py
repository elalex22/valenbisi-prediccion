"""Construcción del dataset limpio: lee los zips, limpia, guarda y resume."""

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from valenbisi.lectura import leer_zip, zips_en_orden
from valenbisi.limpieza import REGLAS_GLOBALES, REGLAS_POR_TROZO, aplicar_reglas, marcar_llena

COLUMNAS_SALIDA = ["ts_utc", "ts", "estacion", "bicis", "libres", "total", "fecha_actualizacion", "llena"]
COLUMNAS_ESTACION = ["direccion", "lat", "lon", "total"]


def resumir_estaciones(conteos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por estación con el valor más frecuente de cada atributo.

    ``conteos`` tiene ``estacion``, :data:`COLUMNAS_ESTACION` y ``n`` (número de registros).
    """
    columnas = []
    for col in COLUMNAS_ESTACION:
        moda = (
            conteos.groupby(["estacion", col])["n"].sum().reset_index()
            .sort_values(["n", col], ascending=[False, True])
            .drop_duplicates("estacion")
            .set_index("estacion")[col]
        )
        columnas.append(moda)
    return pd.concat(columnas, axis=1).sort_index().reset_index()


def construir_dataset(
    carpeta_raw: Path,
    carpeta_salida: Path,
    progreso: Callable[[int, int], None] | None = None,
) -> dict:
    """Lee todos los zips, aplica las reglas de limpieza y escribe en ``carpeta_salida``:

    - ``valenbisi_limpio.parquet``: una fila por (foto, estación) limpia.
    - ``estaciones.parquet``: dirección, coordenadas y capacidad de cada estación.
    - ``resumen_limpieza.json``: el resumen que devuelve esta función.
    """
    zips = zips_en_orden(carpeta_raw)
    conteo: dict[str, list[int]] = {}
    info_total = {"fotos": 0, "fotos_vacias": 0, "fotos_hora_repetida": 0, "fotos_hora_inexistente": 0}
    trozos, conteos_estacion = [], []

    for i, ruta in enumerate(zips, start=1):
        df, info = leer_zip(ruta)
        for clave in info_total:
            info_total[clave] += info[clave]
        if not df.empty:
            conteos_estacion.append(
                df.groupby(["estacion", *COLUMNAS_ESTACION]).size().rename("n").reset_index()
            )
            df = aplicar_reglas(df, REGLAS_POR_TROZO, conteo)
            trozos.append(df[[c for c in COLUMNAS_SALIDA if c != "llena"]])
        if progreso:
            progreso(i, len(zips))

    if not trozos:
        raise ValueError(f"No hay fotos con datos en {carpeta_raw}")
    datos = pd.concat(trozos, ignore_index=True)
    del trozos
    datos = datos.sort_values(["ts_utc", "estacion"], ignore_index=True)
    datos = aplicar_reglas(datos, REGLAS_GLOBALES, conteo)
    datos = marcar_llena(datos).reset_index(drop=True)[COLUMNAS_SALIDA]

    estaciones = resumir_estaciones(pd.concat(conteos_estacion, ignore_index=True))
    estaciones = estaciones[estaciones["estacion"].isin(datos["estacion"].unique())]

    resumen = {
        "zips": len(zips),
        **info_total,
        "reglas": [
            {"regla": nombre, "antes": antes, "eliminadas": antes - despues, "despues": despues}
            for nombre, (antes, despues) in conteo.items()
        ],
        "periodo_local": [str(datos["ts"].min()), str(datos["ts"].max())],
        "fotos_finales": int(datos["ts_utc"].nunique()),
        "estaciones": int(datos["estacion"].nunique()),
        "filas": len(datos),
        "porcentaje_llena": round(float(datos["llena"].mean()) * 100, 2),
    }

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    datos.to_parquet(carpeta_salida / "valenbisi_limpio.parquet", index=False)
    estaciones.to_parquet(carpeta_salida / "estaciones.parquet", index=False)
    (carpeta_salida / "resumen_limpieza.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resumen


def _n(numero: int) -> str:
    """Entero con punto como separador de miles (25.000.000)."""
    return f"{numero:,}".replace(",", ".")


def formatear_resumen(resumen: dict) -> str:
    """Tabla de texto con las filas antes y después de cada regla, y las cifras finales."""
    lineas = [
        f"Zips leídos: {resumen['zips']}   Fotos: {_n(resumen['fotos'])}   "
        f"Fotos vacías (sin filas): {_n(resumen['fotos_vacias'])}",
        f"Cambio de hora: {resumen['fotos_hora_repetida']} fotos de la hora repetida de octubre, "
        f"{resumen['fotos_hora_inexistente']} fotos en hora inexistente de marzo (desplazadas)",
        "",
        f"{'Regla':<32}{'Antes':>14}{'Eliminadas':>14}{'%':>7}{'Después':>14}",
        "-" * 81,
    ]
    for paso in resumen["reglas"]:
        pct = paso["eliminadas"] / paso["antes"] * 100 if paso["antes"] else 0.0
        lineas.append(
            f"{paso['regla']:<32}{_n(paso['antes']):>14}{_n(paso['eliminadas']):>14}"
            f"{pct:>6.1f}%{_n(paso['despues']):>14}"
        )
    desde, hasta = resumen["periodo_local"]
    lineas += [
        "",
        f"Periodo (hora local): {desde} → {hasta}",
        f"Fotos finales: {_n(resumen['fotos_finales'])}   Estaciones: {resumen['estaciones']}   "
        f"Filas: {_n(resumen['filas'])}",
        f"Registros 'llena': {str(resumen['porcentaje_llena']).replace('.', ',')} %",
    ]
    return "\n".join(lineas)
