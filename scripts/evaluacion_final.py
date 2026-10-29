"""Evaluación final en la prueba (mar–sep 2025). Se ejecuta UNA vez.

Decisiones fijadas antes de mirar la prueba (CLAUDE.md, commit 26b0483):
- Principal: LightGBM con FEATURES_MODELO y PARAMETROS, reentrenado con
  entrenamiento + validación, con 1020 árboles fijos (el best_iteration del modelo
  oficial) y sin parada temprana. Las referencias se ajustan con los mismos datos.
- Secundario: el modelo oficial (solo entrenamiento, models/lightgbm.txt), con las
  referencias ajustadas solo con entrenamiento.

Orden, para que nada se ajuste después de ver la prueba:
1. Referencias de los dos escenarios (columnas ligeras de entrenamiento + validación).
2. Modelo principal, guardado en models/lightgbm_final.txt.
3. Modelo secundario cargado y comprobado.
4. Solo entonces se lee la prueba, una vez. Antes de predecir con LightGBM se
   comprueba que la persistencia y la frecuencia del escenario secundario dan las
   cifras de prueba ya publicadas (resumen_referencias.json): si no, la lectura
   está mal y se para.

Mide los dos escenarios (todas, llena ahora / no, tramos de libres), intervalos
al 95 % con bootstrap por días, calibración por tramos y por mes, y las filas de
estaciones nuevas. Guarda data/processed/resumen_prueba.json y las figuras
reports/figures/*_prueba.png.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\evaluacion_final.py --confirmar-prueba
"""

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve

from valenbisi.dataset import formatear_entero
from valenbisi.evaluacion import (
    FRONTERAS,
    TRAMOS_LIBRES,
    ZONA,
    calibracion_por_mes,
    evaluar,
    evaluar_por_tramos,
    intervalos_bootstrap,
    limites,
    limites_union,
    metricas,
    tabla_calibracion,
)
from valenbisi.figuras import COLORES, GRIS, TEXTO, TEXTO_SUAVE, estilo_ejes, guardar
from valenbisi.memoria import comprobar_memoria, evitar_suspension, memoria_libre_gb, memoria_proceso_gb
from valenbisi.modelo import FEATURES_MODELO, PARAMETROS, dataset, entrenar_fijo, matriz, vigilar_memoria
from valenbisi.referencias import (
    ajustar_frecuencia,
    ajustar_libres,
    ajustar_persistencia,
    predecir_frecuencia,
    predecir_libres,
    predecir_persistencia,
)

RAIZ = Path(__file__).resolve().parents[1]
# Fijados antes de mirar la prueba.
ARBOLES_FINALES = 1020
FILAS_ESPERADAS = {"entrenamiento": 10_889_038, "validacion": 4_269_607, "prueba": 5_262_368}
REPLICAS_BOOTSTRAP = 200
MIN_POSITIVOS_ESTACIONES_NUEVAS = 30

COLUMNAS_REFERENCIAS = ["slot_utc", "estacion", "llena", "libres", "llena_60"]
ESCENARIOS = {"principal": "entrenamiento + validación", "secundario": "solo entrenamiento"}
MODELOS = ("constante", "persistencia", "frecuencia", "libres", "lightgbm")
NOMBRES = {"constante": "Constante", "persistencia": "Persistencia", "frecuencia": "Frecuencia histórica",
           "libres": "Tabla por libres", "lightgbm": "LightGBM"}


def memoria() -> str:
    proceso, libre = memoria_proceso_gb(), memoria_libre_gb()
    return "" if proceso is None else f"proceso {proceso[0]:.1f} GB (pico {proceso[1]:.1f}), equipo {libre:.1f} GB libres"


def estado_git() -> dict:
    """Commit con el que se ejecuta y si había cambios sin guardar, para poder rastrear el resultado."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "arbol_limpio": git("status", "--porcelain") == ""}


def comprobar_filas(nombre: str, filas: int) -> None:
    if filas != FILAS_ESPERADAS[nombre]:
        raise ValueError(f"{nombre}: {filas} filas, se esperaban {FILAS_ESPERADAS[nombre]}")


def ajustar_referencias(df: pd.DataFrame) -> dict:
    return {"constante": float(df["llena_60"].mean()), "persistencia": ajustar_persistencia(df),
            "frecuencia": ajustar_frecuencia(df), "libres": ajustar_libres(df)}


def predecir_referencias(referencias: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "constante": np.full(len(df), referencias["constante"]),
        "persistencia": predecir_persistencia(referencias["persistencia"], df),
        "frecuencia": predecir_frecuencia(referencias["frecuencia"], df)[0],
        "libres": predecir_libres(referencias["libres"], df),
    }


def comprobar_referencias_publicadas(prueba: pd.DataFrame, predicciones: dict, p_constante: float,
                                     ruta: Path) -> dict:
    """Las referencias de solo entrenamiento deben dar en la prueba lo que ya se publicó."""
    publicadas = json.loads(ruta.read_text(encoding="utf-8"))["prueba"]["todas"]
    comprobacion = {}
    for modelo in ("persistencia", "frecuencia"):
        ahora = metricas(prueba["llena_60"], predicciones[modelo], p_constante)
        for clave in ("filas", "auc_pr", "brier_skill"):
            if abs(ahora[clave] - publicadas[modelo][clave]) > 1e-4:
                raise ValueError(f"{modelo} {clave}: {ahora[clave]} en la prueba leída, "
                                 f"{publicadas[modelo][clave]} publicado. La lectura de la prueba no cuadra.")
        comprobacion[modelo] = {"auc_pr": ahora["auc_pr"], "brier_skill": ahora["brier_skill"]}
    return comprobacion


def figura_calibracion(y: np.ndarray, predicciones: dict[str, np.ndarray], ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0.0005, 1], [0.0005, 1], color=GRIS, linestyle="--", linewidth=1)
    ax.text(0.5, 0.42, "calibración perfecta", color=TEXTO_SUAVE, fontsize=9, rotation=40, ha="center")
    series = (("libres", COLORES["libres"], "-", 7, "Tabla por libres"),
              ("lightgbm_solo_entrenamiento", COLORES["lightgbm"], ":", 5, "LightGBM (solo entrenamiento)"),
              ("lightgbm", COLORES["lightgbm"], "-", 7, "LightGBM (final)"),
              ("persistencia", COLORES["persistencia"], "none", 10, "Persistencia"))
    for modelo, color, linea, tamano, etiqueta in series:
        tabla = tabla_calibracion(y, predicciones[modelo], n_tramos=20)
        ax.plot(tabla["p_media"], tabla["frecuencia_real"], color=color, linewidth=2, linestyle=linea, marker="o",
                markersize=tamano, markeredgecolor="white", markeredgewidth=1.5, label=etiqueta)
    estilo_ejes(ax)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.0005, 1), ax.set_ylim(0.0005, 1)
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
    for modelo in ("persistencia", "frecuencia", "libres", "lightgbm"):
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


def figura_mensual(mensual: pd.DataFrame, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(1, color=GRIS, linestyle="--", linewidth=1)
    for modelo, etiqueta in (("libres", "Tabla por libres"), ("lightgbm", "LightGBM (final)")):
        ax.plot(mensual["mes"], mensual[f"real_entre_predicho_{modelo}"], color=COLORES[modelo], linewidth=2,
                marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=etiqueta)
    estilo_ejes(ax)
    ax.set_ylabel("Real ÷ predicho (>1: se llenó más de lo previsto)")
    ax.set_title("Calibración por mes en prueba", loc="left", color=TEXTO)
    ax.legend(frameon=False, loc="best")
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
    parser.add_argument("--confirmar-prueba", action="store_true",
                        help="obligatorio: confirma que se quiere leer la prueba (solo se hace una vez)")
    parser.add_argument("--features", type=Path, default=RAIZ / "data" / "processed" / "valenbisi_features.parquet")
    parser.add_argument("--modelo-oficial", type=Path, default=RAIZ / "models" / "lightgbm.txt")
    parser.add_argument("--modelo-final", type=Path, default=RAIZ / "models" / "lightgbm_final.txt")
    parser.add_argument("--figuras", type=Path, default=RAIZ / "reports" / "figures")
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5,
                        help="RAM libre del equipo por debajo de la cual el script se para")
    args = parser.parse_args()
    if not args.confirmar_prueba:
        sys.exit("La evaluación final lee la prueba y solo se hace una vez. Si es el momento, "
                 "vuelve a ejecutarlo con --confirmar-prueba.")
    evitar_suspension()
    inicio = time.monotonic()
    git = estado_git()
    comprobar_memoria(args.minimo_libre_gb)
    print(f"Commit {git['commit'][:7]} (árbol {'limpio' if git['arbol_limpio'] else 'CON CAMBIOS'}); {memoria()}",
          flush=True)

    # 1. Referencias de los dos escenarios, con columnas ligeras.
    union = limites_union(["entrenamiento", "validacion"])
    ligeras = pq.read_table(args.features, columns=COLUMNAS_REFERENCIAS, filters=union).to_pandas()
    solo_entrenamiento = ligeras["slot_utc"] < FRONTERAS["validacion"]
    comprobar_filas("entrenamiento", int(solo_entrenamiento.sum()))
    comprobar_filas("validacion", int((~solo_entrenamiento).sum()))
    referencias = {"principal": ajustar_referencias(ligeras),
                   "secundario": ajustar_referencias(ligeras[solo_entrenamiento])}
    estaciones_conocidas = set(ligeras["estacion"].unique())
    del ligeras, solo_entrenamiento
    gc.collect()
    print(f"Referencias ajustadas ({memoria()})", flush=True)

    # 2. Modelo principal: entrenamiento + validación, árboles fijos.
    tabla = pq.read_table(args.features, columns=[*FEATURES_MODELO, "llena_60"], filters=union)
    if tabla.num_rows != FILAS_ESPERADAS["entrenamiento"] + FILAS_ESPERADAS["validacion"]:
        raise ValueError(f"Entrenamiento + validación: {tabla.num_rows} filas, no cuadra con lo esperado")
    X, y = matriz(tabla), tabla.column("llena_60").to_numpy()
    del tabla
    comprobar_memoria(args.minimo_libre_gb)
    datos = dataset(X, y)
    del X, y
    gc.collect()
    print(f"Entrenando el modelo final: {ARBOLES_FINALES} árboles con "
          f"{formatear_entero(FILAS_ESPERADAS['entrenamiento'] + FILAS_ESPERADAS['validacion'])} filas ({memoria()})",
          flush=True)
    inicio_entrenamiento = time.monotonic()
    final = entrenar_fijo(datos, ARBOLES_FINALES, PARAMETROS, callbacks=[vigilar_memoria(args.minimo_libre_gb)])
    segundos_entrenamiento = time.monotonic() - inicio_entrenamiento
    del datos
    gc.collect()
    if final.num_trees() != ARBOLES_FINALES:
        raise ValueError(f"El modelo final tiene {final.num_trees()} árboles")
    args.modelo_final.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(args.modelo_final)
    print(f"  listo en {segundos_entrenamiento:.0f} s ({memoria()})", flush=True)

    # 3. Modelo secundario: el oficial, sin tocar.
    oficial = lgb.Booster(model_file=str(args.modelo_oficial))
    if oficial.feature_name() != FEATURES_MODELO or oficial.num_trees() != ARBOLES_FINALES:
        raise ValueError("El modelo oficial no tiene las 15 features y los 1020 árboles esperados")

    # 4. La prueba, leída una sola vez.
    print("Leyendo la prueba", flush=True)
    tabla = pq.read_table(args.features, columns=[*FEATURES_MODELO, "slot_utc", "llena_60"],
                          filters=limites("prueba"))
    comprobar_filas("prueba", tabla.num_rows)
    prueba = tabla.select(COLUMNAS_REFERENCIAS).to_pandas()
    X_prueba = matriz(tabla)
    del tabla
    comprobar_memoria(args.minimo_libre_gb)
    y_prueba = prueba["llena_60"].to_numpy()

    predicciones = {e: predecir_referencias(referencias[e], prueba) for e in ESCENARIOS}
    comprobacion = comprobar_referencias_publicadas(
        prueba, predicciones["secundario"], referencias["secundario"]["constante"],
        args.features.with_name("resumen_referencias.json"))
    print(f"Comprobación con las referencias publicadas: correcta {comprobacion}".replace(".", ","), flush=True)

    def predecir(modelo: lgb.Booster) -> np.ndarray:
        return modelo.predict(X_prueba, num_threads=PARAMETROS["num_threads"])

    predicciones["principal"]["lightgbm"] = predecir(final)
    predicciones["secundario"]["lightgbm"] = predecir(oficial)
    del X_prueba
    gc.collect()

    # 5. Métricas de los dos escenarios.
    resultados = {}
    for escenario in ESCENARIOS:
        p_constante = referencias[escenario]["constante"]
        resultados[escenario] = evaluar(prueba, predicciones[escenario], p_constante)
        resultados[escenario].update(evaluar_por_tramos(prueba, predicciones[escenario], p_constante))

    # 6. Intervalos con bootstrap por días. Todos los Brier skill usan la constante
    #    del escenario principal, para que las diferencias sean comparables.
    para_bootstrap = {"lightgbm": predicciones["principal"]["lightgbm"],
                      "libres": predicciones["principal"]["libres"],
                      "lightgbm_solo_entrenamiento": predicciones["secundario"]["lightgbm"]}
    dias = prueba["slot_utc"].dt.tz_convert(ZONA).dt.strftime("%Y-%m-%d").to_numpy()
    print(f"Bootstrap: {REPLICAS_BOOTSTRAP} réplicas de {len(np.unique(dias))} días (unos 35 min)", flush=True)
    inicio_bootstrap = time.monotonic()
    intervalos = intervalos_bootstrap(
        y_prueba, para_bootstrap, dias, referencias["principal"]["constante"],
        diferencias=[("lightgbm", "libres"), ("lightgbm", "lightgbm_solo_entrenamiento")],
        n=REPLICAS_BOOTSTRAP, semilla=0)
    print(f"  listo en {time.monotonic() - inicio_bootstrap:.0f} s ({memoria()})", flush=True)

    # 7. Calibración por tramos y por mes.
    calibracion = {m: tabla_calibracion(y_prueba, p, n_tramos=20).round(5).to_dict("records")
                   for m, p in para_bootstrap.items()}
    mensual = calibracion_por_mes(prueba["slot_utc"], y_prueba,
                                  {m: para_bootstrap[m] for m in ("lightgbm", "libres")}).round(4)

    # 8. Estaciones que no estaban en entrenamiento + validación.
    nuevas = ~prueba["estacion"].isin(estaciones_conocidas).to_numpy()
    estaciones_nuevas = {"estaciones": int(prueba.loc[nuevas, "estacion"].nunique()), "filas": int(nuevas.sum()),
                         "positivos": int(y_prueba[nuevas].sum())}
    if estaciones_nuevas["positivos"] >= MIN_POSITIVOS_ESTACIONES_NUEVAS:
        estaciones_nuevas["metricas"] = {
            m: metricas(y_prueba[nuevas], predicciones["principal"][m][nuevas], referencias["principal"]["constante"])
            for m in ("libres", "lightgbm")}

    todas = {**predicciones["principal"], "lightgbm_solo_entrenamiento": predicciones["secundario"]["lightgbm"]}
    figura_calibracion(y_prueba, todas, args.figuras / "calibracion_prueba.png")
    figura_pr(y_prueba, predicciones["principal"], resultados["principal"]["todas"],
              args.figuras / "curvas_pr_prueba.png")
    figura_mensual(mensual, args.figuras / "mensual_prueba.png")

    proceso = memoria_proceso_gb()
    resumen = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "git": git,
        "decisiones": {"arboles_finales": ARBOLES_FINALES, "features": FEATURES_MODELO, "parametros": PARAMETROS,
                       "escenarios": ESCENARIOS, "replicas_bootstrap": REPLICAS_BOOTSTRAP},
        "filas": FILAS_ESPERADAS,
        "constante": {e: round(referencias[e]["constante"], 5) for e in ESCENARIOS},
        "comprobacion_referencias_publicadas": comprobacion,
        "prueba": resultados,
        "intervalos_95": intervalos,
        "nota_intervalos": "Bootstrap por días locales; todos los Brier skill con la constante del escenario "
                           "principal.",
        "calibracion": calibracion,
        "calibracion_por_mes": mensual.to_dict("records"),
        "estaciones_nuevas": estaciones_nuevas,
        "segundos": {"entrenamiento": round(segundos_entrenamiento), "total": round(time.monotonic() - inicio)},
        "memoria_pico_gb": round(proceso[1], 2) if proceso else None,
    }
    salida = args.features.with_name("resumen_prueba.json")
    salida.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    grupos = [("todas", "Todas las filas"), ("llena_ahora", "Llena ahora"), ("no_llena_ahora", "No llena ahora"),
              *((t, f"Libres = {t}") for t in TRAMOS_LIBRES)]
    for escenario, descripcion in ESCENARIOS.items():
        print(f"\n== Prueba, escenario {escenario} ({descripcion}) ==")
        print(formatear(resultados[escenario], grupos))
    print("\nIntervalos al 95 % (bootstrap por días):")
    for nombre, medidas in intervalos.items():
        for metrica, v in medidas.items():
            print(f"  {nombre:<36}{metrica:<12}{v['estimacion']:>8.4f}  [{v['inferior']:.4f}, {v['superior']:.4f}]"
                  .replace(".", ","))
    print("\nCalibración por mes (real ÷ predicho):")
    print(mensual[["mes", "filas", "frecuencia_real", "real_entre_predicho_lightgbm", "real_entre_predicho_libres"]]
          .to_string(index=False))
    print(f"\nEstaciones nuevas: {estaciones_nuevas}")
    print(f"\nMemoria al acabar: {memoria()}")
    print(f"Guardado en {salida}, {args.modelo_final} y {args.figuras} ({resumen['segundos']['total']} s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
