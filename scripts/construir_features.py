"""Calcula las features de cada fila de data/processed/valenbisi_objetivo.parquet.

El pasado (las fotos de t−15, t−30 y t−60) se busca en valenbisi_limpio.parquet,
que tiene también las fotos sin etiqueta. El objetivo se procesa por bloques
para no tener las 20 M filas con todas sus features en memoria a la vez. Antes
de cada bloque se comprueba la RAM libre y el script se para si baja del mínimo.

Guarda data/processed/valenbisi_features.parquet y resumen_features.json.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\construir_features.py
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from valenbisi.dataset import EscritorPorBloques, formatear_entero
from valenbisi.features import FEATURES, RETARDOS_MIN, construir_features, preparar_referencia
from valenbisi.memoria import comprobar_memoria, memoria_libre_gb, memoria_proceso_gb
from valenbisi.objetivo import alinear_a_slot
from valenbisi.tiempo import VARIABLES, leer_tiempo

RAIZ = Path(__file__).resolve().parents[1]
PROCESADOS = RAIZ / "data" / "processed"
COLUMNAS_OBJETIVO = ["slot_utc", "estacion", "bicis", "libres", "total", "llena", "llena_60"]
CON_NAN = [*(f"delta_libres_{k}" for k in RETARDOS_MIN), "ocupacion", *(f"{v}_ayer" for v in VARIABLES)]


def memoria() -> str:
    proceso, libre = memoria_proceso_gb(), memoria_libre_gb()
    if proceso is None:
        return ""
    return f"proceso {proceso[0]:.1f} GB (pico {proceso[1]:.1f}), equipo {libre:.1f} GB libres"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limpio", type=Path, default=PROCESADOS / "valenbisi_limpio.parquet")
    parser.add_argument("--objetivo", type=Path, default=PROCESADOS / "valenbisi_objetivo.parquet")
    parser.add_argument("--estaciones", type=Path, default=PROCESADOS / "estaciones.parquet")
    parser.add_argument("--tiempo", type=Path, default=RAIZ / "data" / "raw" / "datos_tiempo")
    parser.add_argument("--salida", type=Path, default=PROCESADOS / "valenbisi_features.parquet")
    parser.add_argument("--filas-por-lote", type=int, default=1_000_000)
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5,
                        help="RAM libre del equipo por debajo de la cual el script se para")
    args = parser.parse_args()
    inicio = time.monotonic()
    comprobar_memoria(args.minimo_libre_gb)
    print(f"Al empezar: {memoria()}", flush=True)

    print(f"Leyendo el pasado de {args.limpio}", flush=True)
    limpio = pd.read_parquet(args.limpio, columns=["ts_utc", "estacion", "libres"])
    limpio["slot_utc"] = alinear_a_slot(limpio.pop("ts_utc"))
    referencia = preparar_referencia(limpio)
    filas_referencia = len(limpio)
    del limpio
    gc.collect()
    print(f"  {formatear_entero(filas_referencia)} fotos ordenadas ({memoria()})", flush=True)

    tiempo = leer_tiempo(args.tiempo)
    estaciones = pd.read_parquet(args.estaciones, columns=["estacion", "lat", "lon"])

    conteo = {"filas": 0, "positivos": 0, "nan": dict.fromkeys(CON_NAN, 0)}
    escritor = EscritorPorBloques(args.salida, args.filas_por_lote)
    try:
        lotes = pq.ParquetFile(args.objetivo).iter_batches(batch_size=args.filas_por_lote, columns=COLUMNAS_OBJETIVO)
        for i, lote in enumerate(lotes, start=1):
            comprobar_memoria(args.minimo_libre_gb)
            features = construir_features(lote.to_pandas(), referencia, tiempo, estaciones)
            conteo["filas"] += len(features)
            conteo["positivos"] += int(features["llena_60"].sum())
            for columna in CON_NAN:
                conteo["nan"][columna] += int(features[columna].isna().sum())
            escritor.agregar(features)
            if i % 5 == 0:
                print(f"  {i} lotes, {formatear_entero(conteo['filas'])} filas "
                      f"({time.monotonic() - inicio:.0f} s; {memoria()})", flush=True)
        escritor.cerrar()
    except BaseException:
        escritor.abortar()
        raise

    proceso = memoria_proceso_gb()
    resumen = {
        "filas": conteo["filas"],
        "fotos_de_referencia": filas_referencia,
        "porcentaje_positivos": round(conteo["positivos"] / conteo["filas"] * 100, 2),
        "features": FEATURES,
        "porcentaje_nan": {c: round(n / conteo["filas"] * 100, 3) for c, n in conteo["nan"].items()},
        "memoria_pico_gb": round(proceso[1], 2) if proceso else None,
        "segundos": round(time.monotonic() - inicio),
    }
    salida_resumen = args.salida.with_name("resumen_features.json")
    salida_resumen.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nFilas: {formatear_entero(resumen['filas'])}  positivos: {resumen['porcentaje_positivos']} %")
    print("% de NaN por columna (solo las que pueden tenerlo):")
    for columna, pct in resumen["porcentaje_nan"].items():
        print(f"  {columna:<18}{pct:>8.3f} %".replace(".", ","))
    print(f"\nMemoria al acabar: {memoria()}")
    print(f"Guardado en {args.salida} y {salida_resumen} ({resumen['segundos']} s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
