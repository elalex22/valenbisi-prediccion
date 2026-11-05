"""Evaluación final de la v2 en la prueba v2. Es el único script que lee el clon, y se ejecuta UNA vez.

Decisiones fijadas antes de mirar la prueba v2 (CLAUDE.md, etiqueta v2-congelado):
- LightGBM v2: las 23 features de FEATURES_V2_FINALES con 1557 árboles fijos.
- LightGBM v1: las 15 de FEATURES_MODELO con 1510 árboles fijos.
- Los dos, con PARAMETROS y reentrenados con entrenamiento + validación v2.
- Referencias (constante, persistencia, frecuencia y tabla por libres) con esos mismos datos.
- Resultado principal: todas las filas. Secundario: sin la primera semana (desde el 06-07-2026).

Orden, para que nada se ajuste después de ver la prueba:
1. Referencias, con los datos de desarrollo.
2. Los dos modelos, guardados en disco. Si ya existen con sus features y árboles, se cargan.
3. Solo entonces se abre el clon y se preparan sus filas, con la misma cadena del desarrollo.
4. Métricas, intervalos al 95 % (bootstrap por días, 200 réplicas), calibración y estaciones nuevas.

Se niega a ejecutarse con cambios sin guardar, fuera de v2-congelado o si ya hay
un resumen. Si hay un error de código, se corrige, se repite con --repetir-por-error
y se documenta. Antes de ejecutarlo: scripts/ensayo_evaluacion_v2.py.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\evaluacion_final_v2.py --confirmar-prueba
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
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve

from valenbisi.dataset import formatear_entero
from valenbisi.evaluacion import (
    FIN_PRUEBA_V2,
    FRONTERAS_V2,
    INICIO_SECUNDARIO_V2,
    TRAMOS_LIBRES,
    calibracion_por_mes,
    evaluar,
    evaluar_por_tramos,
    intervalos_bootstrap,
    limites,
    limites_union,
    metricas,
    tabla_calibracion,
)
from valenbisi.experimento import codigos_de_dia
from valenbisi.features_v2 import FEATURES_RETARDOS, FEATURES_V2_FINALES
from valenbisi.figuras import COLORES, GRIS, TEXTO, TEXTO_SUAVE, estilo_ejes, guardar
from valenbisi.memoria import comprobar_memoria, evitar_suspension, memoria_libre_gb, memoria_proceso_gb
from valenbisi.modelo import (
    FEATURES_MODELO,
    PARAMETROS,
    SecuenciaParquet,
    dataset_por_lotes,
    entrenar_fijo,
    predecir_por_lotes,
    vigilar_memoria,
)
from valenbisi.prueba_v2 import preparar_filas
from valenbisi.referencias import (
    ajustar_frecuencia,
    ajustar_libres,
    ajustar_persistencia,
    predecir_frecuencia,
    predecir_libres,
    predecir_persistencia,
)
from valenbisi.seleccion import es_clara

RAIZ = Path(__file__).resolve().parents[1]
PROCESADOS = RAIZ / "data" / "processed"
ETIQUETA = "v2-congelado"

# Fijados antes de mirar la prueba v2.
ARBOLES = {"lightgbm_v2": 1557, "lightgbm_v1": 1510}
COLUMNAS = {"lightgbm_v2": FEATURES_V2_FINALES, "lightgbm_v1": FEATURES_MODELO}
FILAS_ESPERADAS = {"entrenamiento": 15_158_645, "validacion": 5_262_368}
REPLICAS_BOOTSTRAP = 200
MIN_POSITIVOS_ESTACIONES_NUEVAS = 30
RUIDO_SEMILLA = 0.000064  # el de A5/A5b en validación

COLUMNAS_REFERENCIAS = ["slot_utc", "estacion", "llena", "libres", "llena_60"]
MODELOS = ("constante", "persistencia", "frecuencia", "libres", "lightgbm_v1", "lightgbm_v2")
NOMBRES = {"constante": "Constante", "persistencia": "Persistencia", "frecuencia": "Frecuencia histórica",
           "libres": "Tabla por libres", "lightgbm_v1": "LightGBM v1", "lightgbm_v2": "LightGBM v2"}
ESCENARIOS = {"principal": "todas las filas", "secundario": "sin la primera semana"}
MEDIDAS = ("auc_pr", "brier_skill", "logloss")


def memoria() -> str:
    proceso, libre = memoria_proceso_gb(), memoria_libre_gb()
    return "" if proceso is None else f"proceso {proceso[0]:.1f} GB (pico {proceso[1]:.1f}), equipo {libre:.1f} GB libres"


def estado_git() -> dict:
    """Commit con el que se ejecuta, si había cambios sin guardar y si parte de la etiqueta de la congelación."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True)
    return {"commit": git("rev-parse", "HEAD").stdout.strip(),
            "arbol_limpio": git("status", "--porcelain").stdout.strip() == "",
            "desde_congelado": git("merge-base", "--is-ancestor", ETIQUETA, "HEAD").returncode == 0}


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


def obtener_modelo(nombre: str, ruta: Path, features: Path, minimo_libre_gb: float) -> tuple[lgb.Booster, int | None]:
    """El modelo ya guardado si tiene sus features y árboles; si no, lo entrena con entrenamiento + validación."""
    columnas, arboles = COLUMNAS[nombre], ARBOLES[nombre]
    if ruta.exists():
        modelo = lgb.Booster(model_file=str(ruta))
        if modelo.feature_name() != columnas or modelo.num_trees() != arboles:
            raise ValueError(f"{ruta} no tiene las {len(columnas)} features y los {arboles} árboles esperados")
        print(f"{nombre}: cargado de {ruta}", flush=True)
        return modelo, None

    comprobar_memoria(minimo_libre_gb)
    secuencias = [SecuenciaParquet(features, columnas, limites(c, FRONTERAS_V2)) for c in FILAS_ESPERADAS]
    y = np.concatenate([s.leer(["llena_60"])["llena_60"].to_numpy() for s in secuencias])
    if len(y) != sum(FILAS_ESPERADAS.values()):
        raise ValueError(f"Entrenamiento + validación: {len(y)} filas, se esperaban {sum(FILAS_ESPERADAS.values())}")
    print(f"{nombre}: {arboles} árboles con {len(columnas)} features y {formatear_entero(len(y))} filas ({memoria()})",
          flush=True)
    inicio = time.monotonic()
    datos = dataset_por_lotes(secuencias, y)
    modelo = entrenar_fijo(datos, arboles, PARAMETROS, callbacks=[vigilar_memoria(minimo_libre_gb)])
    segundos = round(time.monotonic() - inicio)
    del datos
    gc.collect()
    if modelo.num_trees() != arboles:
        raise ValueError(f"{nombre} tiene {modelo.num_trees()} árboles")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    modelo.save_model(ruta)
    print(f"  listo en {segundos} s ({memoria()})", flush=True)
    return modelo, segundos


def medir(prueba: pd.DataFrame, predicciones: dict[str, np.ndarray], p_constante: float) -> dict:
    """Métricas de cada modelo en todas las filas y por subgrupos."""
    resultado = evaluar(prueba, predicciones, p_constante)
    resultado.update(evaluar_por_tramos(prueba, predicciones, p_constante))
    y = prueba["llena_60"].to_numpy()
    con_retardos = prueba[FEATURES_RETARDOS].notna().all(axis=1).to_numpy()
    for grupo, filtro in (("con_retardos", con_retardos), ("sin_retardos", ~con_retardos)):
        if filtro.any():
            resultado[grupo] = {m: metricas(y[filtro], p[filtro], p_constante) for m, p in predicciones.items()}
    return resultado


def intervalos(prueba: pd.DataFrame, predicciones: dict[str, np.ndarray], p_constante: float) -> dict:
    """Intervalos al 95 % de cada modelo y de la diferencia de LightGBM v2 con cada uno, en las mismas réplicas."""
    comparados = {m: p for m, p in predicciones.items() if m != "constante"}
    diferencias = [("lightgbm_v2", m) for m in comparados if m != "lightgbm_v2"]
    return intervalos_bootstrap(prueba["llena_60"].to_numpy(), comparados, codigos_de_dia(prueba["slot_utc"]),
                                p_constante, diferencias=diferencias, n=REPLICAS_BOOTSTRAP, semilla=0, medidas=MEDIDAS,
                                decimales=6)


def frente_a_v1(intervalo: dict) -> dict:
    """La misma regla de «diferencia clara» que en la validación. Se documenta; no cambia ninguna decisión."""
    diferencia = intervalo["lightgbm_v2 - lightgbm_v1"]["logloss"]
    return {"diferencia_logloss": diferencia, "ruido_semilla": RUIDO_SEMILLA, "umbral": 2 * RUIDO_SEMILLA,
            "mejora_clara": es_clara(diferencia, RUIDO_SEMILLA)}


def estaciones_nuevas(prueba: pd.DataFrame, predicciones: dict[str, np.ndarray], nuevas: list[int],
                      p_constante: float) -> dict:
    filtro = prueba["estacion"].isin(nuevas).to_numpy()
    y = prueba["llena_60"].to_numpy()
    resultado = {"estaciones": nuevas, "filas": int(filtro.sum()), "positivos": int(y[filtro].sum())}
    if resultado["positivos"] >= MIN_POSITIVOS_ESTACIONES_NUEVAS:
        resultado["metricas"] = {m: metricas(y[filtro], predicciones[m][filtro], p_constante)
                                 for m in ("libres", "lightgbm_v1", "lightgbm_v2")}
    return resultado


def figura_calibracion(y: np.ndarray, predicciones: dict[str, np.ndarray], ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0.0005, 1], [0.0005, 1], color=GRIS, linestyle="--", linewidth=1)
    ax.text(0.5, 0.42, "calibración perfecta", color=TEXTO_SUAVE, fontsize=9, rotation=40, ha="center")
    series = (("libres", COLORES["libres"], "-"), ("lightgbm_v1", COLORES["lightgbm"], ":"),
              ("lightgbm_v2", COLORES["lightgbm"], "-"))
    for modelo, color, linea in series:
        tabla = tabla_calibracion(y, predicciones[modelo], n_tramos=20)
        ax.plot(tabla["p_media"], tabla["frecuencia_real"], color=color, linewidth=2, linestyle=linea, marker="o",
                markersize=6, markeredgecolor="white", markeredgewidth=1.5, label=NOMBRES[modelo])
    estilo_ejes(ax)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.0005, 1), ax.set_ylim(0.0005, 1)
    ax.set_xlabel("Probabilidad predicha (media del tramo)")
    ax.set_ylabel("Frecuencia real de llena a t+60")
    ax.set_title("Calibración en la prueba v2 (escala logarítmica)", loc="left", color=TEXTO)
    ax.legend(frameon=False, loc="upper left")
    guardar(fig, ruta)


def figura_pr(y: np.ndarray, predicciones: dict[str, np.ndarray], resultados: dict, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    prevalencia = y.mean()
    ax.axhline(prevalencia, color=GRIS, linestyle="--", linewidth=1)
    ax.text(0.15, prevalencia + 0.015, f"azar = {prevalencia:.3f}".replace(".", ","), color=TEXTO_SUAVE, fontsize=9)
    series = (("persistencia", COLORES["persistencia"], "-"), ("frecuencia", COLORES["frecuencia"], "-"),
              ("libres", COLORES["libres"], "-"), ("lightgbm_v1", COLORES["lightgbm"], ":"),
              ("lightgbm_v2", COLORES["lightgbm"], "-"))
    for modelo, color, linea in series:
        precision, recall, _ = precision_recall_curve(y, predicciones[modelo])
        ax.plot(recall, precision, color=color, linewidth=2, linestyle=linea, drawstyle="steps-post",
                label=f"{NOMBRES[modelo]} (AUC-PR {resultados[modelo]['auc_pr']:.3f})".replace(".", ","))
    estilo_ejes(ax)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("Recall (de las llenas, cuántas detecta)")
    ax.set_ylabel("Precisión (de las avisadas, cuántas se llenan)")
    ax.set_title("Curva precisión-recall en la prueba v2", loc="left", color=TEXTO)
    ax.legend(frameon=False, loc="upper right")
    guardar(fig, ruta)


def figura_mensual(mensual: pd.DataFrame, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(1, color=GRIS, linestyle="--", linewidth=1)
    for modelo, color, linea in (("libres", COLORES["libres"], "-"), ("lightgbm_v1", COLORES["lightgbm"], ":"),
                                 ("lightgbm_v2", COLORES["lightgbm"], "-")):
        ax.plot(mensual["mes"], mensual[f"real_entre_predicho_{modelo}"], color=color, linewidth=2, linestyle=linea,
                marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=NOMBRES[modelo])
    estilo_ejes(ax)
    ax.set_ylabel("Real ÷ predicho (>1: se llenó más de lo previsto)")
    ax.set_title("Calibración por mes en la prueba v2", loc="left", color=TEXTO)
    ax.legend(frameon=False, loc="best")
    guardar(fig, ruta)


def formatear(resultados: dict, grupos: list[tuple[str, str]]) -> str:
    def decimal(valor: float, cifras: int, ancho: int) -> str:
        return f"{valor:>{ancho}.{cifras}f}".replace(".", ",")

    lineas = []
    for grupo, titulo in grupos:
        if grupo not in resultados:
            continue
        lineas += ["", titulo, f"{'Modelo':<22}{'Filas':>12}{'% pos.':>9}{'AUC-PR':>9}{'Skill':>9}{'Logloss':>11}"]
        for m in MODELOS:
            r = resultados[grupo][m]
            lineas.append(f"{NOMBRES[m]:<22}{formatear_entero(r['filas']):>12}{decimal(r['porcentaje_positivos'], 2, 9)}"
                          f"{decimal(r['auc_pr'], 4, 9)}{decimal(r['brier_skill'], 4, 9)}{decimal(r['logloss'], 6, 11)}")
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirmar-prueba", action="store_true",
                        help="obligatorio: confirma que se quiere leer la prueba v2 (solo se hace una vez)")
    parser.add_argument("--repetir-por-error", action="store_true",
                        help="permite repetir tras un error de código, aunque ya haya un resumen; hay que documentarlo")
    parser.add_argument("--clon", type=Path, default=RAIZ / "data" / "raw" / "valenbici-git")
    parser.add_argument("--features", type=Path, default=PROCESADOS / "valenbisi_features_v2.parquet")
    parser.add_argument("--limpio", type=Path, default=PROCESADOS / "valenbisi_limpio.parquet")
    parser.add_argument("--estaciones", type=Path, default=PROCESADOS / "estaciones.parquet")
    parser.add_argument("--carpeta", type=Path, default=PROCESADOS / "prueba_v2",
                        help="carpeta propia para las filas de la prueba v2")
    parser.add_argument("--salida", type=Path, default=PROCESADOS / "resumen_prueba_v2.json")
    parser.add_argument("--modelos", type=Path, default=RAIZ / "models" / "final_v2")
    parser.add_argument("--figuras", type=Path, default=RAIZ / "reports" / "figures")
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5,
                        help="RAM libre del equipo por debajo de la cual el script se para")
    args = parser.parse_args()
    if not args.confirmar_prueba:
        sys.exit("La evaluación final lee la prueba v2 y solo se hace una vez. Si es el momento, "
                 "vuelve a ejecutarlo con --confirmar-prueba.")
    if args.salida.exists() and not args.repetir_por_error:
        sys.exit(f"Ya existe {args.salida}: la prueba v2 ya se evaluó. Solo se repite por un error de código, "
                 "con --repetir-por-error, y se documenta.")
    git = estado_git()
    if not git["arbol_limpio"] or not git["desde_congelado"]:
        sys.exit(f"Hace falta el árbol limpio y un commit posterior a {ETIQUETA}: {git}")
    evitar_suspension()
    inicio = time.monotonic()
    comprobar_memoria(args.minimo_libre_gb)
    print(f"Commit {git['commit'][:7]}; {memoria()}", flush=True)

    # 1. Referencias, solo con los datos de desarrollo.
    ligeras = pq.read_table(args.features, columns=COLUMNAS_REFERENCIAS,
                            filters=limites_union(list(FILAS_ESPERADAS), FRONTERAS_V2)).to_pandas()
    if len(ligeras) != sum(FILAS_ESPERADAS.values()):
        raise ValueError(f"Entrenamiento + validación: {len(ligeras)} filas, se esperaban {sum(FILAS_ESPERADAS.values())}")
    referencias = ajustar_referencias(ligeras)
    del ligeras
    gc.collect()
    print(f"Referencias ajustadas ({memoria()})", flush=True)

    # 2. Los modelos, antes de abrir el clon.
    modelos, segundos = {}, {}
    for nombre in ARBOLES:
        modelos[nombre], segundos[nombre] = obtener_modelo(nombre, args.modelos / f"{nombre}.txt", args.features,
                                                           args.minimo_libre_gb)
        gc.collect()

    # 3. La prueba v2: solo ahora se abre el clon.
    print("Abriendo el clon y preparando la prueba v2", flush=True)
    conocidas = pd.read_parquet(args.estaciones)
    ultimo_pasado = pc.max(pq.read_table(args.limpio, columns=["ts_utc"]).column("ts_utc")).as_py()
    datos = preparar_filas(args.clon, args.carpeta, conocidas, FRONTERAS_V2["prueba"], FIN_PRUEBA_V2,
                           ultimo_pasado=pd.Timestamp(ultimo_pasado), minimo_libre_gb=args.minimo_libre_gb)
    print(f"  {formatear_entero(datos['filas'])} filas, {datos['estaciones']} estaciones "
          f"({len(datos['estaciones_nuevas'])} nuevas) ({memoria()})", flush=True)

    secuencia = SecuenciaParquet(args.carpeta / "features.parquet", FEATURES_V2_FINALES, [])
    prueba = secuencia.leer(["slot_utc", "estacion", "llena", "libres", "llena_60", *FEATURES_RETARDOS])
    predicciones = predecir_referencias(referencias, prueba)
    predicciones["lightgbm_v2"] = predecir_por_lotes(modelos["lightgbm_v2"], secuencia)
    # La secuencia lee cada modelo con sus columnas; las de la v1 son parte de las de la v2.
    predicciones["lightgbm_v1"] = predecir_por_lotes(
        modelos["lightgbm_v1"], SecuenciaParquet(args.carpeta / "features.parquet", FEATURES_MODELO, []))
    predicciones = {m: predicciones[m] for m in MODELOS}
    del modelos
    gc.collect()

    # 4. Métricas e intervalos de los dos resultados.
    p_constante = referencias["constante"]
    filtros = {"principal": np.ones(len(prueba), dtype=bool),
               "secundario": (prueba["slot_utc"] >= INICIO_SECUNDARIO_V2).to_numpy()}
    resultados, intervalos_95, frente = {}, {}, {}
    for escenario, filtro in filtros.items():
        sub = prueba[filtro].reset_index(drop=True)
        sub_p = {m: p[filtro] for m, p in predicciones.items()}
        resultados[escenario] = medir(sub, sub_p, p_constante)
        print(f"Bootstrap del resultado {escenario}: {REPLICAS_BOOTSTRAP} réplicas ({memoria()})", flush=True)
        inicio_bootstrap = time.monotonic()
        intervalos_95[escenario] = intervalos(sub, sub_p, p_constante)
        frente[escenario] = frente_a_v1(intervalos_95[escenario])
        print(f"  listo en {time.monotonic() - inicio_bootstrap:.0f} s", flush=True)

    # 5. Calibración, estaciones nuevas y figuras.
    y = prueba["llena_60"].to_numpy()
    calibrados = ("libres", "lightgbm_v1", "lightgbm_v2")
    calibracion = {m: tabla_calibracion(y, predicciones[m], n_tramos=20).round(5).to_dict("records") for m in calibrados}
    mensual = calibracion_por_mes(prueba["slot_utc"], y, {m: predicciones[m] for m in calibrados}).round(4)
    nuevas = estaciones_nuevas(prueba, predicciones, datos["estaciones_nuevas"], p_constante)

    figura_calibracion(y, predicciones, args.figuras / "calibracion_prueba_v2.png")
    figura_pr(y, predicciones, resultados["principal"]["todas"], args.figuras / "curvas_pr_prueba_v2.png")
    figura_mensual(mensual, args.figuras / "mensual_prueba_v2.png")

    proceso = memoria_proceso_gb()
    resumen = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "git": git,
        "repetida_por_error": args.repetir_por_error,
        "decisiones": {"arboles": ARBOLES, "features": COLUMNAS, "parametros": PARAMETROS, "escenarios": ESCENARIOS,
                       "replicas_bootstrap": REPLICAS_BOOTSTRAP},
        "filas_desarrollo": FILAS_ESPERADAS,
        "constante": round(p_constante, 5),
        "datos": datos,
        "prueba": resultados,
        "intervalos_95": intervalos_95,
        "frente_a_v1": frente,
        "calibracion": calibracion,
        "calibracion_por_mes": mensual.to_dict("records"),
        "estaciones_nuevas": nuevas,
        "segundos": {"entrenamiento": segundos, "total": round(time.monotonic() - inicio)},
        "memoria_pico_gb": round(proceso[1], 2) if proceso else None,
    }
    args.salida.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    grupos = [("todas", "Todas las filas"), ("llena_ahora", "Llena ahora"), ("no_llena_ahora", "No llena ahora"),
              *((t, f"Libres = {t}") for t in TRAMOS_LIBRES), ("con_retardos", "Con los 4 retardos"),
              ("sin_retardos", "Con algún retardo NaN")]
    for escenario, descripcion in ESCENARIOS.items():
        print(f"\n== Prueba v2, resultado {escenario} ({descripcion}) ==")
        print(formatear(resultados[escenario], grupos))
        print("\nIntervalos al 95 % de la diferencia (LightGBM v2 − otro; en logloss, negativo = v2 mejor):")
        for nombre, medidas in intervalos_95[escenario].items():
            if " - " in nombre:
                for metrica, v in medidas.items():
                    print(f"  {nombre:<28}{metrica:<12}{v['estimacion']:>+10.6f}  "
                          f"[{v['inferior']:+.6f}, {v['superior']:+.6f}]".replace(".", ","))
        print(f"  ¿Mejora clara frente a la v1?: {'sí' if frente[escenario]['mejora_clara'] else 'no'}")
    print("\nCalibración por mes (real ÷ predicho):")
    print(mensual[["mes", "filas", "frecuencia_real", *(f"real_entre_predicho_{m}" for m in calibrados)]]
          .to_string(index=False))
    print(f"\nEstaciones nuevas: { {k: v for k, v in nuevas.items() if k != 'metricas'} }")
    print(f"\nMemoria al acabar: {memoria()}")
    print(f"Guardado en {args.salida}, {args.modelos} y {args.figuras} ({resumen['segundos']['total']} s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
