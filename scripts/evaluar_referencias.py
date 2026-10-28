"""Mide las dos referencias (persistencia y frecuencia histórica) en validación y prueba.

Reparte data/processed/valenbisi_objetivo.parquet por fechas, ajusta las
referencias solo con entrenamiento y guarda:
- data/processed/resumen_referencias.json (métricas y tablas de calibración);
- reports/figures/calibracion_referencias.png y curvas_pr_referencias.png (prueba).

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\evaluar_referencias.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from valenbisi.dataset import formatear_entero
from valenbisi.evaluacion import evaluar, repartir, tabla_calibracion
from valenbisi.referencias import ajustar_frecuencia, ajustar_persistencia, predecir_frecuencia, predecir_persistencia

RAIZ = Path(__file__).resolve().parents[1]
COLUMNAS = ["slot_utc", "estacion", "llena", "llena_60"]
MODELOS = ("constante", "persistencia", "frecuencia")
NOMBRES = {"constante": "Constante", "persistencia": "Persistencia", "frecuencia": "Frecuencia histórica"}
# Paleta categórica por defecto (azul, naranja); gris para las líneas de referencia.
COLORES = {"persistencia": "#2a78d6", "frecuencia": "#eb6834"}
GRIS, TEXTO, TEXTO_SUAVE = "#8f8e8a", "#0b0b0b", "#52514e"


def figura_calibracion(y: np.ndarray, predicciones: dict[str, np.ndarray], ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0.005, 1], [0.005, 1], color=GRIS, linestyle="--", linewidth=1)
    ax.text(0.5, 0.42, "calibración perfecta", color=TEXTO_SUAVE, fontsize=9, rotation=40, ha="center")
    # La persistencia solo predice dos valores: puntos sueltos, sin línea que sugiera valores intermedios.
    for modelo, linea, tamano in (("frecuencia", "-", 7), ("persistencia", "none", 10)):
        tabla = tabla_calibracion(y, predicciones[modelo], n_tramos=20)
        ax.plot(tabla["p_media"], tabla["frecuencia_real"], color=COLORES[modelo], linewidth=2, linestyle=linea,
                marker="o", markersize=tamano, markeredgecolor="white", markeredgewidth=1.5, label=NOMBRES[modelo])
    estilo_ejes(ax)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.005, 1), ax.set_ylim(0.005, 1)
    ax.set_xlabel("Probabilidad predicha (media del tramo)")
    ax.set_ylabel("Frecuencia real de llena a t+60")
    ax.set_title("Calibración en prueba (escala logarítmica)", loc="left", color=TEXTO)
    ax.legend(frameon=False, loc="upper left")
    guardar(fig, ruta)


def figura_pr(y: np.ndarray, predicciones: dict[str, np.ndarray], resultados: dict, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    prevalencia = y.mean()
    ax.axhline(prevalencia, color=GRIS, linestyle="--", linewidth=1)
    ax.text(0.15, prevalencia + 0.015, f"azar = {prevalencia:.3f}".replace(".", ","), color=TEXTO_SUAVE, fontsize=9)
    for modelo in ("persistencia", "frecuencia"):
        precision, recall, _ = precision_recall_curve(y, predicciones[modelo])
        ax.plot(recall, precision, color=COLORES[modelo], linewidth=2, drawstyle="steps-post",
                label=f"{NOMBRES[modelo]} (AUC-PR {resultados[modelo]['auc_pr']:.3f})".replace(".", ","))
    estilo_ejes(ax)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("Recall (de las llenas, cuántas detecta)")
    ax.set_ylabel("Precisión (de las avisadas, cuántas se llenan)")
    ax.set_title("Curva precisión-recall en prueba", loc="left", color=TEXTO)
    ax.legend(frameon=False, loc="upper right")
    guardar(fig, ruta)


def estilo_ejes(ax) -> None:
    ax.grid(color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)
    ax.tick_params(colors=TEXTO_SUAVE)


def guardar(fig, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def formatear(resumen: dict, conjunto: str) -> str:
    def decimal(valor: float, cifras: int, ancho: int) -> str:
        return f"{valor:>{ancho}.{cifras}f}".replace(".", ",")

    def fila(nombre: str, m: dict) -> str:
        return (f"{nombre:<22}{formatear_entero(m['filas']):>12}{decimal(m['porcentaje_positivos'], 2, 9)}"
                f"{decimal(m['auc_pr'], 4, 9)}{decimal(m['brier'], 5, 10)}{decimal(m['brier_skill'], 4, 9)}")

    lineas = [f"== {conjunto.capitalize()} =="]
    for grupo, titulo in (("todas", "Todas las filas"), ("llena_ahora", "Llena ahora"),
                          ("no_llena_ahora", "No llena ahora")):
        lineas += ["", titulo, f"{'Modelo':<22}{'Filas':>12}{'% pos.':>9}{'AUC-PR':>9}{'Brier':>10}{'Skill':>9}"]
        lineas += [fila(NOMBRES[m], resumen[conjunto][grupo][m]) for m in MODELOS]
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objetivo", type=Path, default=RAIZ / "data" / "processed" / "valenbisi_objetivo.parquet")
    parser.add_argument("--figuras", type=Path, default=RAIZ / "reports" / "figures")
    args = parser.parse_args()
    inicio = time.monotonic()

    print(f"Leyendo {args.objetivo}", flush=True)
    conjuntos, reparto = repartir(pd.read_parquet(args.objetivo, columns=COLUMNAS))
    entrenamiento = conjuntos.pop("entrenamiento")

    p_constante = float(entrenamiento["llena_60"].mean())
    persistencia = ajustar_persistencia(entrenamiento)
    frecuencia = ajustar_frecuencia(entrenamiento)
    del entrenamiento

    resumen = {
        "reparto": reparto,
        "ajuste": {
            "constante": round(p_constante, 5),
            "persistencia": {"llena_ahora": round(persistencia[True], 5), "no_llena_ahora": round(persistencia[False], 5)},
            "frecuencia_celdas": len(frecuencia.celdas),
        },
    }
    for nombre, conjunto in conjuntos.items():
        p_frecuencia, respaldo = predecir_frecuencia(frecuencia, conjunto)
        predicciones = {
            "constante": np.full(len(conjunto), p_constante),
            "persistencia": predecir_persistencia(persistencia, conjunto),
            "frecuencia": p_frecuencia,
        }
        resumen[nombre] = evaluar(conjunto, predicciones, p_constante)
        resumen[nombre]["filas_con_respaldo"] = int(respaldo.sum())
        y = conjunto["llena_60"].to_numpy()
        resumen[nombre]["calibracion"] = {
            modelo: tabla_calibracion(y, predicciones[modelo], n_tramos=20).round(5).to_dict("records")
            for modelo in ("persistencia", "frecuencia")
        }
        if nombre == "prueba":
            figura_calibracion(y, predicciones, args.figuras / "calibracion_referencias.png")
            figura_pr(y, predicciones, resumen[nombre]["todas"], args.figuras / "curvas_pr_referencias.png")

    salida = args.objetivo.with_name("resumen_referencias.json")
    salida.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nFilas: {reparto['filas']}  descartadas por el hueco: {reparto['descartadas']}")
    print(f"Persistencia (entrenamiento): llena ahora {persistencia[True]:.4f}, no llena {persistencia[False]:.4f}")
    print(f"Frecuencia: {len(frecuencia.celdas)} celdas; filas con respaldo en prueba: "
          f"{resumen['prueba']['filas_con_respaldo']}\n")
    print(formatear(resumen, "validacion"), end="\n\n")
    print(formatear(resumen, "prueba"))
    print(f"\nGuardado en {salida} y {args.figuras} ({time.monotonic() - inicio:.0f} s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
