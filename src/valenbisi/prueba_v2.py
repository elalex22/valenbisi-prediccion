"""Filas de un periodo nuevo, con la misma cadena que el desarrollo: limpieza, etiqueta y features.

Lo usan la evaluación final de la v2 (con el clon) y su ensayo (con un trozo de
la validación). Todo se escribe en una carpeta propia: los Parquet de desarrollo
no se tocan, así que sus salvaguardas contra la prueba v2 siguen valiendo.

Aquí no se calcula ninguna métrica ni se resume la etiqueta: solo se preparan datos.
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from valenbisi.dataset import EscritorPorBloques, construir_dataset
from valenbisi.evaluacion import ZONA
from valenbisi.features import preparar_referencia
from valenbisi.features_v2 import FEATURES_V2_FINALES, features_finales_v2
from valenbisi.memoria import comprobar_memoria
from valenbisi.objetivo import HORIZONTE, alinear_a_slot, construir_objetivo
from valenbisi.vecinas import K_VECINAS, preparar_rejilla, vecinas_mas_cercanas

COLUMNAS_OBJETIVO = ["slot_utc", "estacion", "bicis", "libres", "total", "llena", "llena_60"]
COLUMNAS_ESTACION = ["estacion", "lat", "lon"]
# El retardo más largo de las features: si el pasado de desarrollo está más lejos, no hace falta.
RETARDO_MAXIMO = pd.Timedelta(days=7)


def sin_zona(momento: pd.Timestamp) -> pd.Timestamp:
    """Un instante como hora local sin zona, el formato de ``ts``."""
    return momento.tz_convert(ZONA).tz_localize(None)


def estaciones_y_vecinas(conocidas: pd.DataFrame, vistas: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, object]:
    """Ubicaciones de todas las estaciones, las nuevas y la tabla de vecinas.

    Las conocidas conservan las coordenadas de desarrollo. Una estación nueva busca
    vecinas entre las conocidas, pero no pasa a ser vecina de ninguna: así las
    conocidas tienen las mismas vecinas con las que se entrenó el modelo.
    """
    conocidas = conocidas[COLUMNAS_ESTACION]
    nuevas = vistas.loc[~vistas["estacion"].isin(conocidas["estacion"]), COLUMNAS_ESTACION]
    todas = pd.concat([conocidas, nuevas], ignore_index=True).sort_values("estacion", ignore_index=True)
    return todas, nuevas["estacion"], vecinas_mas_cercanas(todas, k=K_VECINAS, candidatas=conocidas)


def preparar_filas(raw: Path, carpeta: Path, conocidas: pd.DataFrame, inicio: pd.Timestamp, fin: pd.Timestamp,
                   ultimo_pasado: pd.Timestamp | None = None, filas_por_lote: int = 1_000_000,
                   minimo_libre_gb: float = 1.5) -> dict:
    """Escribe ``carpeta/features.parquet`` con las filas con ``inicio ≤ slot_utc`` y ``slot_utc + 60 min < fin``.

    Solo se buscan retardos entre las fotos del propio periodo. Con ``ultimo_pasado``
    (la última foto de desarrollo) se comprueba que eso no pierde nada: si el periodo
    empieza a menos de 7 días de ella, se para.
    """
    carpeta = Path(carpeta)
    limpieza = construir_dataset(raw, carpeta, rango_local=(sin_zona(inicio), sin_zona(fin)))
    construir_objetivo(carpeta / "valenbisi_limpio.parquet", carpeta / "valenbisi_objetivo.parquet")

    limpio = pd.read_parquet(carpeta / "valenbisi_limpio.parquet", columns=["ts_utc", "estacion", "libres", "bicis"])
    limpio["slot_utc"] = alinear_a_slot(limpio.pop("ts_utc"))
    if ultimo_pasado is not None and limpio["slot_utc"].min() - ultimo_pasado <= RETARDO_MAXIMO:
        raise ValueError(f"Los datos empiezan el {limpio['slot_utc'].min()}, a menos de {RETARDO_MAXIMO} de la última "
                         f"foto de desarrollo ({ultimo_pasado}): harían falta sus fotos como pasado")
    referencia, rejilla = preparar_referencia(limpio), preparar_rejilla(limpio)
    del limpio

    vistas = pd.read_parquet(carpeta / "estaciones.parquet", columns=COLUMNAS_ESTACION)
    estaciones, nuevas, vecinas = estaciones_y_vecinas(conocidas, vistas)

    filas = fuera = nan = 0
    escritor = EscritorPorBloques(carpeta / "features.parquet", filas_por_lote)
    try:
        lotes = pq.ParquetFile(carpeta / "valenbisi_objetivo.parquet").iter_batches(
            batch_size=filas_por_lote, columns=COLUMNAS_OBJETIVO)
        for lote in lotes:
            comprobar_memoria(minimo_libre_gb)
            lote = lote.to_pandas()
            dentro = (lote["slot_utc"] >= inicio) & (lote["slot_utc"] + HORIZONTE < fin)
            fuera += int((~dentro).sum())
            lote = lote[dentro].reset_index(drop=True)
            if lote.empty:
                continue
            features = features_finales_v2(lote, referencia, rejilla, vecinas, estaciones)
            filas += len(features)
            nan += features[FEATURES_V2_FINALES].isna().sum()
            escritor.agregar(features)
        if filas == 0:
            raise ValueError("Ninguna fila queda dentro del periodo")
        escritor.cerrar()
    except BaseException:
        escritor.abortar()
        raise

    return {
        "zips": limpieza["zips"],
        "fotos_finales": limpieza["fotos_finales"],
        "fotos_hora_repetida": limpieza["fotos_hora_repetida"],
        "filas": filas,
        "filas_fuera_del_periodo": fuera,
        "estaciones": limpieza["estaciones"],
        "estaciones_nuevas": sorted(int(e) for e in nuevas),
        "porcentaje_nan": {c: round(int(n) / filas * 100, 3) for c, n in nan.items() if n},
    }
