"""Ensayo de la evaluación final de la v2 con datos ya usados: nunca abre el clon.

Pasa por la cadena de la evaluación (scripts/evaluacion_final_v2.py) un trozo de
valenbici-master que ya es validación v2, como si fuera un periodo nuevo, y
comprueba con datos reales que:
- las 23 features son exactamente las de valenbisi_features_v2.parquet, en las
  filas que ya tienen 7 días de pasado dentro del trozo;
- el modelo A5 da con ellas las predicciones que ya estaban guardadas;
- las decisiones congeladas coinciden con los resúmenes de la ablación y de la comparación.

Si algo no cuadra, se para con un error. No entrena nada.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\ensayo_evaluacion_v2.py
"""

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from valenbisi.evaluacion import FRONTERAS_V2, ZONA, limites
from valenbisi.features_v2 import FEATURES_V2_FINALES
from valenbisi.memoria import comprobar_memoria, evitar_suspension
from valenbisi.modelo import SecuenciaParquet, predecir_por_lotes
from valenbisi.prueba_v2 import RETARDO_MAXIMO, preparar_filas

RAIZ = Path(__file__).resolve().parents[1]
PROCESADOS = RAIZ / "data" / "processed"
sys.path.insert(0, str(RAIZ / "scripts"))
from evaluacion_final_v2 import ARBOLES, COLUMNAS, FILAS_ESPERADAS  # noqa: E402

INICIO = pd.Timestamp("2025-08-01", tz=ZONA).tz_convert("UTC")
# La frontera de la prueba v2: la validación acaba 60 min antes, igual que la cadena con su fin.
FIN = FRONTERAS_V2["prueba"]
TOLERANCIA_PREDICCION = 1e-6  # las predicciones guardadas son float32


def comprobar_decisiones() -> None:
    ablacion = json.loads((PROCESADOS / "resumen_ablacion_v2.json").read_text(encoding="utf-8"))
    comparacion = json.loads((PROCESADOS / "resumen_comparacion_modelos_v2.json").read_text(encoding="utf-8"))
    decision = comparacion["decisiones"]
    errores = {
        "modelo elegido": (decision["configuracion_elegida"], "A5"),
        "árboles de la v2": (ARBOLES["lightgbm_v2"], ablacion["variantes"]["A5"]["arboles"]),
        "árboles de la v1": (ARBOLES["lightgbm_v1"], ablacion["variantes"]["A0"]["arboles"]),
        "features de la v2": (COLUMNAS["lightgbm_v2"], ablacion["variantes"]["A5"]["features"]),
        "features de la v1": (COLUMNAS["lightgbm_v1"], ablacion["variantes"]["A0"]["features"]),
        "filas": (FILAS_ESPERADAS, comparacion["filas"]),
    }
    for nombre, (en_script, en_resumen) in errores.items():
        if en_script != en_resumen:
            raise ValueError(f"{nombre}: {en_script} en el script, {en_resumen} en los resúmenes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAIZ / "data" / "raw" / "valenbici-master")
    parser.add_argument("--carpeta", type=Path, default=PROCESADOS / "ensayo_v2")
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5)
    args = parser.parse_args()
    evitar_suspension()
    comprobar_memoria(args.minimo_libre_gb)
    inicio = time.monotonic()

    comprobar_decisiones()
    print("Decisiones congeladas: coinciden con los resúmenes", flush=True)

    conocidas = pd.read_parquet(PROCESADOS / "estaciones.parquet")
    datos = preparar_filas(args.raw, args.carpeta, conocidas, INICIO, FIN, minimo_libre_gb=args.minimo_libre_gb)
    print(f"Cadena lista: {datos['filas']} filas, {datos['zips']} zips ({time.monotonic() - inicio:.0f} s)", flush=True)
    if datos["estaciones_nuevas"]:
        raise ValueError(f"Estaciones nuevas en datos de desarrollo: {datos['estaciones_nuevas']}")

    # Solo las filas con todo su pasado dentro del trozo: más 60 min, por el t−60 de las vecinas a t−7 días.
    desde = INICIO + RETARDO_MAXIMO + pd.Timedelta(minutes=60)
    orden = ["slot_utc", "estacion"]
    columnas = [*orden, *(c for c in FEATURES_V2_FINALES if c != "estacion"), "llena_60"]
    tabla = pd.read_parquet(args.carpeta / "features.parquet", columns=columnas,
                            filters=[("slot_utc", ">=", desde)]).sort_values(orden, ignore_index=True)
    filtros = [("slot_utc", ">=", desde), ("slot_utc", "<", FIN - pd.Timedelta(minutes=60))]
    original = pd.read_parquet(PROCESADOS / "valenbisi_features_v2.parquet", columns=columnas,
                               filters=filtros).sort_values(orden, ignore_index=True)
    if len(tabla) != len(original):
        raise ValueError(f"{len(tabla)} filas en la cadena, {len(original)} en el Parquet de features")
    for columna in columnas:
        a, b = tabla[columna].to_numpy(), original[columna].to_numpy()
        iguales = (a == b) | (pd.isna(a) & pd.isna(b))
        if not iguales.all():
            raise ValueError(f"{columna}: {int((~iguales).sum())} filas distintas del Parquet de features")
    print(f"Features: {len(tabla)} filas y {len(columnas)} columnas idénticas al Parquet", flush=True)

    # Las mismas predicciones de A5: validación, en el orden del Parquet.
    secuencia = SecuenciaParquet(args.carpeta / "features.parquet", FEATURES_V2_FINALES, [("slot_utc", ">=", desde)])
    nuevas = secuencia.leer(orden).assign(p=predecir_por_lotes(
        lgb.Booster(model_file=str(RAIZ / "models" / "ablacion_v2" / "A5.txt")), secuencia))
    validacion = pq.read_table(PROCESADOS / "valenbisi_features_v2.parquet", columns=orden,
                               filters=limites("validacion", FRONTERAS_V2)).to_pandas()
    validacion["guardada"] = np.load(PROCESADOS / "ablacion_v2" / "A5.npy")
    cruce = nuevas.merge(validacion, on=orden, how="left", validate="one_to_one")
    diferencia = (cruce["p"] - cruce["guardada"]).abs().max()
    if cruce["guardada"].isna().any() or diferencia > TOLERANCIA_PREDICCION:
        raise ValueError(f"Las predicciones de A5 no coinciden: diferencia máxima {diferencia}")
    print(f"Predicciones de A5: {len(cruce)} filas, diferencia máxima {diferencia:.2e}", flush=True)

    # En Windows, la carpeta solo se puede borrar con sus ficheros ya cerrados.
    del secuencia
    gc.collect()
    shutil.rmtree(args.carpeta)
    print(f"\nEnsayo correcto en {time.monotonic() - inicio:.0f} s; carpeta temporal borrada")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
