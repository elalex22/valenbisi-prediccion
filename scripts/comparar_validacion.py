"""Compara LightGBM con las tres referencias en validación, sin reentrenar.

- Carga el modelo guardado en models/lightgbm.txt.
- Ajusta constante, persistencia, frecuencia y tabla por anclajes libres solo con
  entrenamiento, y lee validación con su propio filtro: la prueba no se carga.
- Mide en todo validación, por «llena ahora» y por tramos de anclajes libres.
- Importancia por permutación por grupos de features dentro de los subgrupos
  donde se decide si una estación se llena («no llena ahora» y 1–5 libres).

Guarda data/processed/resumen_comparacion_validacion.json y las figuras
reports/figures/*_comparacion_validacion.png.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\comparar_validacion.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve

from valenbisi.dataset import formatear_entero
from valenbisi.evaluacion import FRONTERAS, TRAMOS_LIBRES, evaluar, evaluar_por_tramos, limites, tabla_calibracion
from valenbisi.features import FEATURES
from valenbisi.figuras import COLORES, GRIS, TEXTO, TEXTO_SUAVE, estilo_ejes, guardar
from valenbisi.memoria import comprobar_memoria, memoria_libre_gb, memoria_proceso_gb
from valenbisi.modelo import GRUPOS, PARAMETROS, importancia_permutacion, matriz
from valenbisi.referencias import (
    ajustar_frecuencia,
    ajustar_libres,
    ajustar_persistencia,
    predecir_frecuencia,
    predecir_libres,
    predecir_persistencia,
)

RAIZ = Path(__file__).resolve().parents[1]
COLUMNAS_REFERENCIAS = ["slot_utc", "estacion", "llena", "libres", "llena_60"]
MODELOS = ("constante", "persistencia", "frecuencia", "libres", "lightgbm")
NOMBRES = {"constante": "Constante", "persistencia": "Persistencia", "frecuencia": "Frecuencia histórica",
           "libres": "Tabla por libres", "lightgbm": "LightGBM"}
FILAS_PERMUTACION = 500_000
# Subgrupos para la permutación: (nombre, función de la fila -> si entra).
SUBGRUPOS_PERMUTACION = {
    "todas": lambda d: np.ones(len(d), dtype=bool),
    "no_llena_ahora": lambda d: ~d["llena"].to_numpy(),
    "libres_1_5": lambda d: d["libres"].between(1, 5).to_numpy(),
}


def memoria() -> str:
    proceso, libre = memoria_proceso_gb(), memoria_libre_gb()
    return "" if proceso is None else f"proceso {proceso[0]:.1f} GB (pico {proceso[1]:.1f}), equipo {libre:.1f} GB libres"


def leer(ruta: Path, conjunto: str, columnas: list[str]):
    tabla = pq.read_table(ruta, columns=columnas, filters=limites(conjunto))
    # Comprobación explícita de que nada de prueba entra en memoria (las lecturas
    # sin slot_utc usan el mismo filtro, y main() comprueba que dan las mismas filas).
    if "slot_utc" in columnas and pc.max(tabla.column("slot_utc")).as_py() >= FRONTERAS["prueba"]:
        raise ValueError(f"La lectura de {conjunto} contiene filas de prueba")
    return tabla


def figura_calibracion(y: np.ndarray, predicciones: dict[str, np.ndarray], ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0.0005, 1], [0.0005, 1], color=GRIS, linestyle="--", linewidth=1)
    ax.text(0.5, 0.42, "calibración perfecta", color=TEXTO_SUAVE, fontsize=9, rotation=40, ha="center")
    for modelo, linea, tamano in (("libres", "-", 7), ("lightgbm", "-", 7), ("persistencia", "none", 10)):
        tabla = tabla_calibracion(y, predicciones[modelo], n_tramos=20)
        ax.plot(tabla["p_media"], tabla["frecuencia_real"], color=COLORES[modelo], linewidth=2, linestyle=linea,
                marker="o", markersize=tamano, markeredgecolor="white", markeredgewidth=1.5, label=NOMBRES[modelo])
    estilo_ejes(ax)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.0005, 1), ax.set_ylim(0.0005, 1)
    ax.set_xlabel("Probabilidad predicha (media del tramo)")
    ax.set_ylabel("Frecuencia real de llena a t+60")
    ax.set_title("Calibración en validación (escala logarítmica)", loc="left", color=TEXTO)
    ax.legend(frameon=False, loc="upper left")
    guardar(fig, ruta)


def figura_pr(y: np.ndarray, predicciones: dict[str, np.ndarray], resultados: dict, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    prevalencia = y.mean()
    ax.axhline(prevalencia, color=GRIS, linestyle="--", linewidth=1)
    ax.text(0.15, prevalencia + 0.015, f"azar = {prevalencia:.3f}".replace(".", ","), color=TEXTO_SUAVE, fontsize=9)
    for modelo in ("persistencia", "frecuencia", "libres", "lightgbm"):
        precision, recall, _ = precision_recall_curve(y, predicciones[modelo])
        ax.plot(recall, precision, color=COLORES[modelo], linewidth=2, drawstyle="steps-post",
                label=f"{NOMBRES[modelo]} (AUC-PR {resultados[modelo]['auc_pr']:.3f})".replace(".", ","))
    estilo_ejes(ax)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("Recall (de las llenas, cuántas detecta)")
    ax.set_ylabel("Precisión (de las avisadas, cuántas se llenan)")
    ax.set_title("Curva precisión-recall en validación", loc="left", color=TEXTO)
    ax.legend(frameon=False, loc="upper right")
    guardar(fig, ruta)


def formatear(resultados: dict, grupos: list[tuple[str, str]]) -> str:
    def decimal(valor: float, cifras: int, ancho: int) -> str:
        return f"{valor:>{ancho}.{cifras}f}".replace(".", ",")

    lineas = []
    for grupo, titulo in grupos:
        lineas += ["", titulo, f"{'Modelo':<22}{'Filas':>12}{'% pos.':>9}{'AUC-PR':>9}{'Brier':>10}{'Skill':>9}"]
        for m in MODELOS:
            r = resultados[grupo][m]
            lineas.append(f"{NOMBRES[m]:<22}{formatear_entero(r['filas']):>12}{decimal(r['porcentaje_positivos'], 2, 9)}"
                          f"{decimal(r['auc_pr'], 4, 9)}{decimal(r['brier'], 5, 10)}{decimal(r['brier_skill'], 4, 9)}")
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=RAIZ / "data" / "processed" / "valenbisi_features.parquet")
    parser.add_argument("--modelo", type=Path, default=RAIZ / "models" / "lightgbm.txt")
    parser.add_argument("--figuras", type=Path, default=RAIZ / "reports" / "figures")
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5,
                        help="RAM libre del equipo por debajo de la cual el script se para")
    args = parser.parse_args()
    inicio = time.monotonic()
    comprobar_memoria(args.minimo_libre_gb)

    # 1. Referencias, ajustadas con entrenamiento.
    entrenamiento = leer(args.features, "entrenamiento", COLUMNAS_REFERENCIAS).to_pandas()
    p_constante = float(entrenamiento["llena_60"].mean())
    persistencia = ajustar_persistencia(entrenamiento)
    frecuencia = ajustar_frecuencia(entrenamiento)
    libres = ajustar_libres(entrenamiento)
    filas_entrenamiento = len(entrenamiento)
    del entrenamiento
    comprobar_memoria(args.minimo_libre_gb)

    validacion = leer(args.features, "validacion", COLUMNAS_REFERENCIAS).to_pandas()
    predicciones = {
        "constante": np.full(len(validacion), p_constante),
        "persistencia": predecir_persistencia(persistencia, validacion),
        "frecuencia": predecir_frecuencia(frecuencia, validacion)[0],
        "libres": predecir_libres(libres, validacion),
    }

    # 2. LightGBM guardado, sobre las mismas filas de validación.
    modelo = lgb.Booster(model_file=str(args.modelo))
    if modelo.feature_name() != FEATURES:
        raise ValueError("Las features del modelo guardado no coinciden con FEATURES")
    tabla = leer(args.features, "validacion", [*FEATURES, "llena_60"])
    X_val, y_val = matriz(tabla), tabla.column("llena_60").to_numpy()
    del tabla
    if len(X_val) != len(validacion) or not (X_val[:, FEATURES.index("libres")] == validacion["libres"]).all():
        raise ValueError("Las filas de validación no coinciden entre las dos lecturas")
    comprobar_memoria(args.minimo_libre_gb)
    print(f"Validación: {formatear_entero(len(X_val))} filas ({memoria()})", flush=True)

    def predecir(X: np.ndarray) -> np.ndarray:
        return modelo.predict(X, num_threads=PARAMETROS["num_threads"])

    predicciones["lightgbm"] = predecir(X_val)
    resultados = evaluar(validacion, predicciones, p_constante)
    resultados.update(evaluar_por_tramos(validacion, predicciones, p_constante))
    calibracion = {m: tabla_calibracion(y_val, predicciones[m], n_tramos=20).round(5).to_dict("records")
                   for m in ("persistencia", "frecuencia", "libres", "lightgbm")}

    # 3. Permutación por grupos de features dentro de cada subgrupo de filas.
    permutacion = {}
    rng = np.random.default_rng(0)
    for nombre, condicion in SUBGRUPOS_PERMUTACION.items():
        indices = np.flatnonzero(condicion(validacion))
        muestra = rng.choice(indices, size=min(FILAS_PERMUTACION, len(indices)), replace=False)
        print(f"Permutación en {nombre}: {formatear_entero(len(muestra))} filas", flush=True)
        base, caidas = importancia_permutacion(
            predecir, X_val[muestra], y_val[muestra],
            {g: [FEATURES.index(f) for f in cols] for g, cols in GRUPOS.items()}, semilla=1)
        permutacion[nombre] = {"filas": len(muestra), "positivos": round(float(y_val[muestra].mean()), 4),
                               "auc_pr_sin_barajar": round(base, 4),
                               "por_grupo": {g: round(c, 4) for g, c in caidas.items()}}
        comprobar_memoria(args.minimo_libre_gb)

    figura_calibracion(y_val, predicciones, args.figuras / "calibracion_comparacion_validacion.png")
    figura_pr(y_val, predicciones, resultados["todas"], args.figuras / "curvas_pr_comparacion_validacion.png")

    proceso = memoria_proceso_gb()
    resumen = {
        "filas": {"entrenamiento": filas_entrenamiento, "validacion": len(X_val)},
        "ajuste_libres": {"celdas": len(libres), "maximo_libres": int(libres.index.max()),
                          "proporcion": {int(k): round(float(v), 5) for k, v in libres.items()}},
        "tramos_libres": {k: list(v) for k, v in TRAMOS_LIBRES.items()},
        "validacion": resultados,
        "calibracion_validacion": calibracion,
        "permutacion_por_subgrupo": permutacion,
        "segundos": round(time.monotonic() - inicio),
        "memoria_pico_gb": round(proceso[1], 2) if proceso else None,
    }
    salida = args.features.with_name("resumen_comparacion_validacion.json")
    salida.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== Validación ==")
    print(formatear(resultados, [("todas", "Todas las filas"), ("llena_ahora", "Llena ahora"),
                                 ("no_llena_ahora", "No llena ahora"),
                                 *((t, f"Libres = {t}") for t in TRAMOS_LIBRES)]))
    print("\nPermutación por grupos (caída de AUC-PR):")
    for nombre, p in permutacion.items():
        print(f"  {nombre} (AUC-PR sin barajar {p['auc_pr_sin_barajar']:.4f}, {p['positivos']:.1%} de positivos)"
              .replace(".", ","))
        for g, c in sorted(p["por_grupo"].items(), key=lambda x: -x[1]):
            print(f"    {g:<22}{c:>8.4f}".replace(".", ","))
    print(f"\nMemoria al acabar: {memoria()}")
    print(f"Guardado en {salida} y {args.figuras} ({resumen['segundos']} s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
