"""Entrena LightGBM con entrenamiento y lo compara con las referencias en validación.

- Lee de data/processed/valenbisi_features.parquet solo las filas de cada
  conjunto (filtros de pyarrow): la prueba no se carga.
- Entrena con entrenamiento; validación solo sirve para la parada temprana y
  para medir. Las referencias se ajustan también solo con entrenamiento.
- Vigila la RAM del equipo y se para si baja del mínimo.

Guarda data/processed/resumen_lightgbm.json, models/lightgbm.txt y las figuras
reports/figures/*_lightgbm_validacion.png.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\entrenar_lightgbm.py
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve

from valenbisi.dataset import formatear_entero
from valenbisi.evaluacion import evaluar, limites, tabla_calibracion
from valenbisi.figuras import COLORES, GRIS, TEXTO, TEXTO_SUAVE, estilo_ejes, guardar
from valenbisi.memoria import comprobar_memoria, evitar_suspension, memoria_libre_gb, memoria_proceso_gb
from valenbisi.modelo import (
    FEATURES_MODELO,
    GRUPOS_MODELO,
    PARAMETROS,
    dataset,
    entrenar,
    importancia_permutacion,
    matriz,
    vigilar_memoria,
)
from valenbisi.referencias import ajustar_frecuencia, ajustar_persistencia, predecir_frecuencia, predecir_persistencia

RAIZ = Path(__file__).resolve().parents[1]
COLUMNAS_REFERENCIAS = ["slot_utc", "estacion", "llena", "llena_60"]
MODELOS = ("constante", "persistencia", "frecuencia", "lightgbm")
NOMBRES = {"constante": "Constante", "persistencia": "Persistencia", "frecuencia": "Frecuencia histórica",
           "lightgbm": "LightGBM"}
FILAS_PERMUTACION = 500_000


def memoria() -> str:
    proceso, libre = memoria_proceso_gb(), memoria_libre_gb()
    return "" if proceso is None else f"proceso {proceso[0]:.1f} GB (pico {proceso[1]:.1f}), equipo {libre:.1f} GB libres"


def leer(ruta: Path, conjunto: str, columnas: list[str]):
    return pq.read_table(ruta, columns=columnas, filters=limites(conjunto))


def figura_calibracion(y: np.ndarray, predicciones: dict[str, np.ndarray], ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0.002, 1], [0.002, 1], color=GRIS, linestyle="--", linewidth=1)
    ax.text(0.5, 0.42, "calibración perfecta", color=TEXTO_SUAVE, fontsize=9, rotation=40, ha="center")
    # La persistencia solo predice dos valores: puntos sueltos, sin línea que sugiera valores intermedios.
    for modelo, linea, tamano in (("frecuencia", "-", 7), ("lightgbm", "-", 7), ("persistencia", "none", 10)):
        tabla = tabla_calibracion(y, predicciones[modelo], n_tramos=20)
        ax.plot(tabla["p_media"], tabla["frecuencia_real"], color=COLORES[modelo], linewidth=2, linestyle=linea,
                marker="o", markersize=tamano, markeredgecolor="white", markeredgewidth=1.5, label=NOMBRES[modelo])
    estilo_ejes(ax)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.002, 1), ax.set_ylim(0.002, 1)
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
    for modelo in ("persistencia", "frecuencia", "lightgbm"):
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


def figura_importancias(ganancia: dict[str, float], permutacion: dict[str, float], ruta: Path) -> None:
    """Dos paneles con el mismo orden de features (el de la permutación), cada uno con su escala."""
    orden = sorted(permutacion, key=permutacion.get)
    fig, (izq, der) = plt.subplots(1, 2, figsize=(10, 6.5), sharey=True)
    for ax, valores, titulo, etiqueta in (
        (izq, permutacion, "Permutación en validación", "Caída de AUC-PR al barajarla"),
        (der, ganancia, "Ganancia en entrenamiento", "% de la reducción total de la pérdida"),
    ):
        ax.barh(orden, [valores[f] for f in orden], color=COLORES["lightgbm"], height=0.7)
        estilo_ejes(ax)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(etiqueta, color=TEXTO_SUAVE)
        ax.set_title(titulo, loc="left", color=TEXTO)
    guardar(fig, ruta)


def formatear(resultados: dict) -> str:
    def decimal(valor: float, cifras: int, ancho: int) -> str:
        return f"{valor:>{ancho}.{cifras}f}".replace(".", ",")

    lineas = []
    for grupo, titulo in (("todas", "Todas las filas"), ("llena_ahora", "Llena ahora"),
                          ("no_llena_ahora", "No llena ahora")):
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
    evitar_suspension()
    inicio = time.monotonic()
    comprobar_memoria(args.minimo_libre_gb)
    print(f"Al empezar: {memoria()}", flush=True)

    # 1. Referencias, ajustadas con entrenamiento (solo 4 columnas ligeras).
    entrenamiento = leer(args.features, "entrenamiento", COLUMNAS_REFERENCIAS).to_pandas()
    p_constante = float(entrenamiento["llena_60"].mean())
    persistencia = ajustar_persistencia(entrenamiento)
    frecuencia = ajustar_frecuencia(entrenamiento)
    del entrenamiento
    validacion = leer(args.features, "validacion", COLUMNAS_REFERENCIAS).to_pandas()
    predicciones = {
        "constante": np.full(len(validacion), p_constante),
        "persistencia": predecir_persistencia(persistencia, validacion),
        "frecuencia": predecir_frecuencia(frecuencia, validacion)[0],
    }

    # 2. Matrices de LightGBM. Cada tabla de Arrow se suelta en cuanto está en la matriz,
    #    y la de entrenamiento en cuanto LightGBM ha construido sus histogramas.
    tabla = leer(args.features, "entrenamiento", [*FEATURES_MODELO, "llena_60"])
    filas_entrenamiento = tabla.num_rows
    X, y = matriz(tabla), tabla.column("llena_60").to_numpy()
    del tabla
    print(f"Entrenamiento: {formatear_entero(filas_entrenamiento)} filas en memoria ({memoria()})", flush=True)
    comprobar_memoria(args.minimo_libre_gb)
    d_entrenamiento = dataset(X, y)
    del X, y
    gc.collect()
    print(f"  histogramas construidos ({memoria()})", flush=True)

    tabla = leer(args.features, "validacion", [*FEATURES_MODELO, "llena_60"])
    X_val, y_val = matriz(tabla), tabla.column("llena_60").to_numpy()
    del tabla
    if len(X_val) != len(validacion) or not (X_val[:, FEATURES_MODELO.index("llena")] == validacion["llena"]).all():
        raise ValueError("Las filas de validación no coinciden entre las dos lecturas")
    d_validacion = dataset(X_val, y_val, referencia=d_entrenamiento)
    comprobar_memoria(args.minimo_libre_gb)
    print(f"Validación: {formatear_entero(len(X_val))} filas ({memoria()})", flush=True)

    # 3. Entrenamiento con parada temprana en validación.
    print("Entrenando (logloss de validación cada 100 árboles)", flush=True)
    inicio_entrenamiento = time.monotonic()
    modelo = entrenar(d_entrenamiento, d_validacion, callbacks=[vigilar_memoria(args.minimo_libre_gb)])
    segundos_entrenamiento = time.monotonic() - inicio_entrenamiento
    mejor = modelo.best_iteration
    print(f"  {mejor} árboles ({segundos_entrenamiento:.0f} s; {memoria()})", flush=True)
    args.modelo.parent.mkdir(parents=True, exist_ok=True)
    modelo.save_model(args.modelo, num_iteration=mejor)
    del d_entrenamiento, d_validacion
    gc.collect()

    # 4. Evaluación en validación.
    def predecir(X: np.ndarray) -> np.ndarray:
        return modelo.predict(X, num_iteration=mejor, num_threads=PARAMETROS["num_threads"])

    predicciones["lightgbm"] = predecir(X_val)
    resultados = evaluar(validacion, predicciones, p_constante)
    calibracion = {m: tabla_calibracion(y_val, predicciones[m], n_tramos=20).round(5).to_dict("records")
                   for m in ("persistencia", "frecuencia", "lightgbm")}

    # 5. Importancias: ganancia (gratis, en entrenamiento) y permutación (en una muestra de validación).
    ganancia_bruta = modelo.feature_importance("gain", iteration=mejor)
    ganancia = {f: round(float(g / ganancia_bruta.sum() * 100), 2) for f, g in zip(FEATURES_MODELO, ganancia_bruta)}
    print(f"Importancia por permutación en {formatear_entero(FILAS_PERMUTACION)} filas de validación", flush=True)
    muestra = np.random.default_rng(0).choice(len(X_val), size=min(FILAS_PERMUTACION, len(X_val)), replace=False)
    X_m, y_m = X_val[muestra], y_val[muestra]
    base, por_feature = importancia_permutacion(predecir, X_m, y_m, {f: [i] for i, f in enumerate(FEATURES_MODELO)})
    grupos = {g: [FEATURES_MODELO.index(f) for f in cols] for g, cols in GRUPOS_MODELO.items()}
    _, por_grupo = importancia_permutacion(predecir, X_m, y_m, grupos, semilla=1)

    figura_calibracion(y_val, predicciones, args.figuras / "calibracion_lightgbm_validacion.png")
    figura_pr(y_val, predicciones, resultados["todas"], args.figuras / "curvas_pr_lightgbm_validacion.png")
    figura_importancias(ganancia, por_feature, args.figuras / "importancias_lightgbm_validacion.png")

    proceso = memoria_proceso_gb()
    resumen = {
        "parametros": PARAMETROS,
        "arboles": mejor,
        "logloss_validacion": round(modelo.best_score["validacion"]["binary_logloss"], 5),
        "filas": {"entrenamiento": filas_entrenamiento, "validacion": len(X_val)},
        "validacion": resultados,
        "calibracion_validacion": calibracion,
        "importancia": {
            "ganancia_porcentaje": ganancia,
            "permutacion_auc_pr": {"filas": len(muestra), "auc_pr_sin_barajar": round(base, 4),
                                   "por_feature": {f: round(c, 4) for f, c in por_feature.items()},
                                   "por_grupo": {g: round(c, 4) for g, c in por_grupo.items()}},
        },
        "segundos": {"entrenamiento": round(segundos_entrenamiento), "total": round(time.monotonic() - inicio)},
        "memoria_pico_gb": round(proceso[1], 2) if proceso else None,
    }
    salida = args.features.with_name("resumen_lightgbm.json")
    salida.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== Validación ==")
    print(formatear(resultados))
    print(f"\nPermutación (caída de AUC-PR; sin barajar {base:.4f}):".replace(".", ","))
    for g, c in sorted(por_grupo.items(), key=lambda x: -x[1]):
        print(f"  {g:<22}{c:>8.4f}".replace(".", ","))
    for f, c in sorted(por_feature.items(), key=lambda x: -x[1]):
        print(f"    {f:<20}{c:>8.4f}   ganancia {ganancia[f]:>6.2f} %".replace(".", ","))
    print(f"\nMemoria al acabar: {memoria()}")
    print(f"Guardado en {salida}, {args.modelo} y {args.figuras} ({resumen['segundos']['total']} s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
