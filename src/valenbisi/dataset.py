"""Construcción del dataset limpio: lee los zips, limpia, guarda y resume."""

import json
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from valenbisi.lectura import leer_zip, zips_en_orden
from valenbisi.limpieza import aplicar_reglas, crear_reglas, marcar_llena

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


class _EscritorPorBloques:
    """Acumula trozos y los escribe en un Parquet en bloques de ``filas_por_bloque``.

    Escribe en un fichero temporal y solo lo renombra al cerrar sin errores,
    para no dejar un Parquet a medias si el proceso se interrumpe.
    """

    def __init__(self, ruta: Path, filas_por_bloque: int) -> None:
        self.ruta = ruta
        self.temporal = ruta.with_name(ruta.name + ".tmp")
        self.filas_por_bloque = filas_por_bloque
        self._pendientes: list[pd.DataFrame] = []
        self._filas_pendientes = 0
        self._escritor: pq.ParquetWriter | None = None

    def agregar(self, df: pd.DataFrame) -> None:
        self._pendientes.append(df)
        self._filas_pendientes += len(df)
        if self._filas_pendientes >= self.filas_por_bloque:
            self._volcar()

    def _volcar(self) -> None:
        if not self._pendientes:
            return
        tabla = pa.Table.from_pandas(pd.concat(self._pendientes, ignore_index=True), preserve_index=False)
        if self._escritor is None:
            self._escritor = pq.ParquetWriter(self.temporal, tabla.schema)
        self._escritor.write_table(tabla)
        self._pendientes, self._filas_pendientes = [], 0

    def cerrar(self) -> None:
        self._volcar()
        if self._escritor is not None:
            self._escritor.close()
            os.replace(self.temporal, self.ruta)

    def abortar(self) -> None:
        if self._escritor is not None:
            self._escritor.close()
        self.temporal.unlink(missing_ok=True)


def construir_dataset(
    carpeta_raw: Path,
    carpeta_salida: Path,
    progreso: Callable[[int, int], None] | None = None,
    filas_por_bloque: int = 500_000,
) -> dict:
    """Lee todos los zips, aplica las reglas de limpieza y escribe en ``carpeta_salida``:

    - ``valenbisi_limpio.parquet``: una fila por (foto, estación) limpia.
    - ``estaciones.parquet``: dirección, coordenadas y capacidad de cada estación.
    - ``resumen_limpieza.json``: el resumen que devuelve esta función.

    Procesa zip a zip en orden cronológico y escribe sobre la marcha, así que
    en memoria solo hay un zip (y un bloque pendiente de escribir) a la vez.
    """
    zips = zips_en_orden(carpeta_raw)
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    reglas = crear_reglas()
    conteo: dict[str, list[int]] = {}
    info_total = {"fotos": 0, "fotos_vacias": 0, "fotos_hora_repetida": 0, "fotos_hora_inexistente": 0}
    conteos_estacion = []
    estaciones_vistas: set[int] = set()
    filas = llenas = fotos_finales = 0
    ts_min = ts_max = None
    ultima_ts_utc = None

    escritor = _EscritorPorBloques(carpeta_salida / "valenbisi_limpio.parquet", filas_por_bloque)
    try:
        for i, ruta in enumerate(zips, start=1):
            df, info = leer_zip(ruta)
            for clave in info_total:
                info_total[clave] += info[clave]
            if not df.empty:
                conteos_estacion.append(
                    df.groupby(["estacion", *COLUMNAS_ESTACION]).size().rename("n").reset_index()
                )
                # Las reglas de fotos idénticas y de duplicados solo miran dentro
                # del trozo (y la última foto anterior): exige que no haya solapes.
                primera, ultima = df["ts_utc"].min(), df["ts_utc"].max()
                if pd.notna(primera):
                    if ultima_ts_utc is not None and primera <= ultima_ts_utc:
                        raise ValueError(
                            f"{ruta.name} trae fotos desde {primera}, anteriores o iguales a la "
                            f"última ya procesada ({ultima_ts_utc}): los zips no están en orden"
                        )
                    ultima_ts_utc = ultima

                df = df.sort_values(["ts_utc", "estacion"], ignore_index=True)
                df = aplicar_reglas(df, reglas, conteo)
                if not df.empty:
                    df = marcar_llena(df)[COLUMNAS_SALIDA]
                    escritor.agregar(df)
                    filas += len(df)
                    llenas += int(df["llena"].sum())
                    fotos_finales += df["ts_utc"].nunique()
                    estaciones_vistas.update(df["estacion"].unique().tolist())
                    ts_min = df["ts"].min() if ts_min is None else min(ts_min, df["ts"].min())
                    ts_max = df["ts"].max() if ts_max is None else max(ts_max, df["ts"].max())
            if progreso:
                progreso(i, len(zips))

        if filas == 0:
            raise ValueError(f"No hay fotos con datos en {carpeta_raw}")
        escritor.cerrar()
    except BaseException:
        escritor.abortar()
        raise

    estaciones = resumir_estaciones(pd.concat(conteos_estacion, ignore_index=True))
    estaciones = estaciones[estaciones["estacion"].isin(estaciones_vistas)]
    estaciones.to_parquet(carpeta_salida / "estaciones.parquet", index=False)

    resumen = {
        "zips": len(zips),
        **info_total,
        "reglas": [
            {"regla": nombre, "antes": antes, "eliminadas": antes - despues, "despues": despues}
            for nombre, (antes, despues) in conteo.items()
        ],
        "periodo_local": [str(ts_min), str(ts_max)],
        "fotos_finales": fotos_finales,
        "estaciones": len(estaciones_vistas),
        "filas": filas,
        "porcentaje_llena": round(llenas / filas * 100, 2),
    }
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
