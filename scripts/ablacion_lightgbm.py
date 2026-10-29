"""Ablación de LightGBM: ¿aportan la tendencia y el tiempo de ayer?

Entrena varias variantes con los mismos parámetros y la misma parada temprana
que el modelo oficial, y las mide en validación:
- el modelo completo con dos semillas (la diferencia entre ellas es el ruido);
- sin la tendencia, sin el tiempo de ayer y sin las dos cosas.
Un grupo de features merece quitarse si quitarlo cambia las métricas menos que
el ruido de semilla.

No sobrescribe el modelo oficial (models/lightgbm.txt) ni su resumen: guarda
data/processed/resumen_ablacion.json, actualizado al acabar cada variante.
Vigila la RAM y se para si baja del mínimo. Mientras corre, pide a Windows que
no suspenda el equipo.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\ablacion_lightgbm.py
"""

import argparse
import ctypes
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from valenbisi.dataset import formatear_entero
from valenbisi.evaluacion import evaluar, evaluar_por_tramos, limites
from valenbisi.features import FEATURES
from valenbisi.memoria import comprobar_memoria, memoria_libre_gb, memoria_proceso_gb
from valenbisi.modelo import GRUPOS, PARAMETROS, dataset, entrenar, matriz, vigilar_memoria

RAIZ = Path(__file__).resolve().parents[1]


def sin(*grupos: str) -> list[str]:
    quitar = {f for g in grupos for f in GRUPOS[g]}
    return [f for f in FEATURES if f not in quitar]


# nombre -> (features, semilla)
VARIANTES = {
    "completo_semilla_42": (FEATURES, 42),
    "completo_semilla_43": (FEATURES, 43),
    "sin_tendencia": (sin("tendencia"), 42),
    "sin_tiempo_ayer": (sin("tiempo de ayer"), 42),
    "sin_tendencia_ni_tiempo": (sin("tendencia", "tiempo de ayer"), 42),
}


def memoria() -> str:
    proceso, libre = memoria_proceso_gb(), memoria_libre_gb()
    return "" if proceso is None else f"proceso {proceso[0]:.1f} GB (pico {proceso[1]:.1f}), equipo {libre:.1f} GB libres"


def evitar_suspension() -> None:
    """Pide a Windows que no suspenda el equipo mientras este proceso siga vivo."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def leer(ruta: Path, conjunto: str, columnas: list[str]):
    return pq.read_table(ruta, columns=columnas, filters=limites(conjunto))


def entrenar_variante(ruta: Path, columnas: list[str], semilla: int, minimo_libre_gb: float):
    """Entrena con entrenamiento y parada temprana en validación; devuelve el modelo y la proporción de positivos."""
    tabla = leer(ruta, "entrenamiento", [*columnas, "llena_60"])
    X, y = matriz(tabla, columnas), tabla.column("llena_60").to_numpy()
    del tabla
    p_constante = float(y.mean())
    comprobar_memoria(minimo_libre_gb)
    d_entrenamiento = dataset(X, y, columnas)
    del X, y
    gc.collect()

    tabla = leer(ruta, "validacion", [*columnas, "llena_60"])
    X_val, y_val = matriz(tabla, columnas), tabla.column("llena_60").to_numpy()
    del tabla
    d_validacion = dataset(X_val, y_val, columnas, referencia=d_entrenamiento)
    del X_val, y_val
    gc.collect()
    comprobar_memoria(minimo_libre_gb)
    print(f"  datos listos ({memoria()})", flush=True)

    modelo = entrenar(d_entrenamiento, d_validacion, parametros={**PARAMETROS, "seed": semilla},
                      callbacks=[vigilar_memoria(minimo_libre_gb)], registro=500)
    del d_entrenamiento, d_validacion
    gc.collect()
    return modelo, p_constante


def formatear(resultados: dict) -> str:
    def decimal(valor: float, cifras: int = 4) -> str:
        return f"{valor:.{cifras}f}".replace(".", ",")

    cabecera = (f"{'Variante':<26}{'Feat.':>6}{'Árboles':>8}{'Logloss':>9}{'AUC-PR':>8}{'Skill':>8}"
                f"{'AP no ll.':>10}{'AP 1':>8}{'AP 2-3':>8}{'AP 4-6':>8}")
    lineas = [cabecera]
    for nombre, r in resultados.items():
        v = r["validacion"]
        lineas.append(
            f"{nombre:<26}{len(r['features']):>6}{r['arboles']:>8}{decimal(r['logloss_validacion'], 5):>9}"
            f"{decimal(v['todas']['lightgbm']['auc_pr']):>8}{decimal(v['todas']['lightgbm']['brier_skill']):>8}"
            f"{decimal(v['no_llena_ahora']['lightgbm']['auc_pr']):>10}{decimal(v['1']['lightgbm']['auc_pr']):>8}"
            f"{decimal(v['2-3']['lightgbm']['auc_pr']):>8}{decimal(v['4-6']['lightgbm']['auc_pr']):>8}")
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=RAIZ / "data" / "processed" / "valenbisi_features.parquet")
    parser.add_argument("--variantes", nargs="+", choices=list(VARIANTES), default=list(VARIANTES))
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5,
                        help="RAM libre del equipo por debajo de la cual el script se para")
    args = parser.parse_args()
    evitar_suspension()
    inicio = time.monotonic()
    comprobar_memoria(args.minimo_libre_gb)
    salida = args.features.with_name("resumen_ablacion.json")

    # Columnas ligeras de validación para medir (llena, libres y la etiqueta).
    validacion = leer(args.features, "validacion", ["llena", "libres", "llena_60"]).to_pandas()

    resultados = {}
    for nombre in args.variantes:
        columnas, semilla = VARIANTES[nombre]
        print(f"\n== {nombre}: {len(columnas)} features, semilla {semilla} ==", flush=True)
        inicio_variante = time.monotonic()
        modelo, p_constante = entrenar_variante(args.features, columnas, semilla, args.minimo_libre_gb)
        segundos_entrenamiento = time.monotonic() - inicio_variante

        tabla = leer(args.features, "validacion", columnas)
        X_val = matriz(tabla, columnas)
        del tabla
        if len(X_val) != len(validacion) or not (X_val[:, columnas.index("libres")] == validacion["libres"]).all():
            raise ValueError("Las filas de validación no coinciden entre las dos lecturas")
        p = modelo.predict(X_val, num_iteration=modelo.best_iteration, num_threads=PARAMETROS["num_threads"])
        del X_val
        predicciones = {"lightgbm": p}
        medidas = evaluar(validacion, predicciones, p_constante)
        medidas.update(evaluar_por_tramos(validacion, predicciones, p_constante))

        proceso = memoria_proceso_gb()
        resultados[nombre] = {
            "features": columnas,
            "semilla": semilla,
            "arboles": modelo.best_iteration,
            "logloss_validacion": round(modelo.best_score["validacion"]["binary_logloss"], 6),
            "validacion": medidas,
            "segundos": round(time.monotonic() - inicio_variante),
            "segundos_entrenamiento": round(segundos_entrenamiento),
            "memoria_pico_gb": round(proceso[1], 2) if proceso else None,
        }
        # Se guarda tras cada variante: si algo se para, lo hecho no se pierde.
        salida.write_text(json.dumps({"parametros": PARAMETROS, "variantes": resultados}, ensure_ascii=False,
                                     indent=2), encoding="utf-8")
        todas = medidas["todas"]["lightgbm"]
        print(f"  {modelo.best_iteration} árboles, AUC-PR {todas['auc_pr']:.4f}, skill {todas['brier_skill']:.4f} "
              f"({resultados[nombre]['segundos']} s; {memoria()})", flush=True)
        del modelo, p, predicciones
        gc.collect()

    print("\n== Validación ==")
    print(formatear(resultados))
    print(f"\nGuardado en {salida} ({time.monotonic() - inicio:.0f} s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
