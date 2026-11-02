"""XGBoost con las 23 features de la v2, como alternativa a LightGBM. Reglas fijadas en CLAUDE.md antes de ejecutar.

Reparto v2: entrenamiento dic-2022 a feb-2025, validación mar–sep 2025. Tres
configuraciones, todas con hist, max_bin 256, eta 0,05, subsample y
colsample_bytree de 0,8, lambda 1, min_child_weight 50 (≈ 1000 filas), la
estación categórica nativa y parada temprana por la logloss de validación
(paciencia de 100, tope de 3000 árboles):

  X1 lossguide, 63 hojas (la de partida: la misma forma que LightGBM)
  X2 lossguide, 255 hojas
  X3 depthwise, profundidad 8

En lossguide se quita el límite de profundidad (max_depth 0, el de XGBoost es 6),
para que el tamaño lo marque solo el número de hojas, como en LightGBM.

La de menor logloss se repite con la semilla 43 para medir el ruido. Sustituye
a X1 solo si mejora más del doble de ese ruido y el intervalo al 95 % (bootstrap
por días) de la diferencia pareada excluye el 0. Si se queda X1 sin ser la de
menor logloss, también se repite X1 con la semilla 43.

Las matrices (QuantileDMatrix por bloques del Parquet) se construyen una vez y
sirven para todas las configuraciones. Guarda las predicciones de validación en
data/processed/modelos_v2/, los modelos (recortados en su mejor árbol) en
models/modelos_v2/ y data/processed/resumen_xgboost_v2.json, actualizado tras
cada paso. Vigila la RAM y evita la suspensión.

Con --reanudar, reutiliza los entrenamientos idénticos (columnas en el mismo
orden, semilla, parámetros y parada temprana) del resumen anterior.

Con --humo, solo mide la memoria con los 2 primeros bloques de entrenamiento y
de validación y 20 árboles de X2, y estima el pico con todas las filas. No
escribe ningún resumen, predicción ni modelo.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\xgboost_v2.py --humo
    .venv\\Scripts\\python scripts\\xgboost_v2.py [--reanudar]
"""

import argparse
import copy
import gc
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from valenbisi.evaluacion import FRONTERAS_V2, limites
from valenbisi.experimento import Experimento, comprobar_sin_prueba_v2
from valenbisi.features_v2 import FEATURES_V2_FINALES
from valenbisi.memoria import comprobar_memoria, evitar_suspension, memoria_libre_gb, memoria_proceso_gb
from valenbisi.modelo import SecuenciaParquet
from valenbisi.modelo_xgboost import PARAMETROS_XGB, VigilarMemoria, entrenar_xgb, matrices, predecir_xgb
from valenbisi.reanudar import clave
from valenbisi.seleccion import elegir_configuracion

RAIZ = Path(__file__).resolve().parents[1]
PROCESADOS = RAIZ / "data" / "processed"
CONFIGURACIONES = {
    "X1": {"grow_policy": "lossguide", "max_leaves": 63, "max_depth": 0},
    "X2": {"grow_policy": "lossguide", "max_leaves": 255, "max_depth": 0},
    "X3": {"grow_policy": "depthwise", "max_depth": 8},
}
PARTIDA = "X1"
MAX_ARBOLES = 3000
PACIENCIA = 100
REPLICAS = 200
HUMO_BLOQUES = 2
HUMO_ARBOLES = 20


def parametros(nombre: str) -> dict:
    return {**PARAMETROS_XGB, **CONFIGURACIONES[nombre]}


def memoria() -> str:
    proceso, libre = memoria_proceso_gb(), memoria_libre_gb()
    return "" if proceso is None else f"proceso {proceso[0]:.1f} GB (pico {proceso[1]:.1f}), equipo {libre:.1f} GB libres"


class Entrenamientos:
    def __init__(self, exp: Experimento, minimo_libre_gb: float):
        self.exp = exp
        self.minimo_libre_gb = minimo_libre_gb
        self.datos: tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix] | None = None

    def preparar(self) -> tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix]:
        if self.datos is None:
            inicio = time.monotonic()
            self.datos = matrices(self.exp.secuencia("entrenamiento"), self.exp.y_ent,
                                  self.exp.secuencia("validacion"), self.exp.y_val, PARAMETROS_XGB["max_bin"])
            print(f"  matrices listas en {time.monotonic() - inicio:.0f} s ({memoria()})", flush=True)
        return self.datos

    def variante(self, nombre: str, config: str, semilla: int) -> None:
        params = parametros(config)
        if self.exp.reutilizar(nombre, clave(self.exp.columnas, semilla, params, MAX_ARBOLES, PACIENCIA)):
            return
        comprobar_memoria(self.minimo_libre_gb)
        print(f"\n== {nombre}: {CONFIGURACIONES[config]}, semilla {semilla} ==", flush=True)
        d_ent, d_val = self.preparar()
        inicio = time.monotonic()
        modelo = entrenar_xgb(d_ent, d_val, {**params, "seed": semilla}, MAX_ARBOLES, PACIENCIA,
                              callbacks=[VigilarMemoria(self.minimo_libre_gb)], registro=500)
        segundos = time.monotonic() - inicio
        rondas = int(modelo.attr("rondas_entrenadas"))
        tope = rondas >= MAX_ARBOLES
        # Se guarda recortado en el mejor árbol y se predice con el modelo guardado: lo que se reutilice es lo medido.
        ruta = self.exp.modelos / f"{nombre}.ubj"
        modelo.save_model(str(ruta))
        del modelo
        modelo = xgb.Booster(model_file=str(ruta))
        modelo.set_param({"nthread": params["nthread"]})
        p = predecir_xgb(modelo, self.exp.secuencia("validacion"))
        self.exp.registrar(nombre, p, {"configuracion": config, "features": self.exp.columnas, "semilla": semilla,
                                       "parametros": params, "max_arboles": MAX_ARBOLES, "paciencia": PACIENCIA,
                                       "arboles": modelo.num_boosted_rounds(), "tope_alcanzado": tope,
                                       "segundos_entrenamiento": round(segundos), "modelo": ruta.name})
        print(f"  {modelo.num_boosted_rounds()} árboles{' (TOPE)' if tope else ''}, logloss "
              f"{self.exp.logloss(nombre):.6f} ({segundos:.0f} s; {memoria()})", flush=True)
        del modelo
        gc.collect()


def recortar(secuencia: SecuenciaParquet, bloques: int) -> SecuenciaParquet:
    """La misma secuencia, pero solo con sus ``bloques`` primeros bloques no vacíos."""
    s = copy.copy(secuencia)
    usados = [g for g, f in enumerate(secuencia.filas_bloque) if len(f)][:bloques]
    s.filas_bloque = [f if g in usados else f[:0] for g, f in enumerate(secuencia.filas_bloque)]
    s.fin = np.cumsum([len(f) for f in s.filas_bloque])
    s.inicio = s.fin - [len(f) for f in s.filas_bloque]
    s._cache = None
    return s


def humo(features: Path, minimo_libre_gb: float) -> dict:
    """Memoria con unos pocos bloques reales y estimación del pico con todas las filas (no escribe nada)."""
    comprobar_sin_prueba_v2(features)
    base = memoria_proceso_gb()[0]
    sec = {c: SecuenciaParquet(features, FEATURES_V2_FINALES, limites(c, FRONTERAS_V2))
           for c in ("entrenamiento", "validacion")}
    total = {c: len(s) for c, s in sec.items()}
    sec = {c: recortar(s, HUMO_BLOQUES) for c, s in sec.items()}
    y = {c: s.leer(["llena_60"])["llena_60"].to_numpy() for c, s in sec.items()}
    usadas = {c: len(s) for c, s in sec.items()}
    inicio = time.monotonic()
    d_ent, d_val = matrices(sec["entrenamiento"], y["entrenamiento"], sec["validacion"], y["validacion"])
    tras_matrices = memoria_proceso_gb()[0]
    print(f"  matrices de {usadas['entrenamiento']:,} + {usadas['validacion']:,} filas en "
          f"{time.monotonic() - inicio:.0f} s ({memoria()})".replace(",", "."), flush=True)
    inicio = time.monotonic()
    xgb.train({**parametros("X2"), "seed": 42}, d_ent, num_boost_round=HUMO_ARBOLES,
              evals=[(d_val, "validacion")], verbose_eval=False, callbacks=[VigilarMemoria(minimo_libre_gb, cada=1)])
    segundos_arbol = (time.monotonic() - inicio) / HUMO_ARBOLES
    pico = memoria_proceso_gb()[1]
    escala = (total["entrenamiento"] + total["validacion"]) / (usadas["entrenamiento"] + usadas["validacion"])
    estimado = base + (pico - base) * escala
    resultado = {"filas_usadas": usadas, "filas_totales": total, "gb_base": round(base, 2),
                 "gb_tras_matrices": round(tras_matrices, 2), "gb_pico": round(pico, 2),
                 "gb_pico_estimado_todas": round(estimado, 2),
                 "segundos_por_arbol_estimado_todas": round(segundos_arbol * total["entrenamiento"]
                                                            / usadas["entrenamiento"], 2)}
    print(f"\nPico con {HUMO_BLOQUES} bloques: {pico:.2f} GB (base {base:.2f}). Estimado con todas las filas "
          f"(lineal, por lo alto): {estimado:.1f} GB; unos {resultado['segundos_por_arbol_estimado_todas']:.1f} s "
          f"por árbol de X2. Equipo: {memoria_libre_gb():.1f} GB libres.", flush=True)
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=PROCESADOS / "valenbisi_features_v2.parquet")
    parser.add_argument("--salida", type=Path, default=PROCESADOS / "resumen_xgboost_v2.json")
    parser.add_argument("--predicciones", type=Path, default=PROCESADOS / "modelos_v2")
    parser.add_argument("--modelos", type=Path, default=RAIZ / "models" / "modelos_v2")
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5,
                        help="RAM libre del equipo por debajo de la cual el script se para")
    parser.add_argument("--reanudar", action="store_true",
                        help="reutiliza los entrenamientos idénticos del resumen anterior")
    parser.add_argument("--humo", action="store_true",
                        help="solo mide la memoria con unos pocos bloques reales; no escribe nada")
    args = parser.parse_args()
    evitar_suspension()
    comprobar_memoria(args.minimo_libre_gb)
    if args.humo:
        humo(args.features, args.minimo_libre_gb)
        return
    inicio = time.monotonic()

    exp = Experimento(args.features, args.salida, args.predicciones, args.modelos, reanudar=args.reanudar,
                      replicas=REPLICAS)
    print(f"Entrenamiento {len(exp.y_ent):,} filas, validación {len(exp.y_val):,} ({memoria()})".replace(",", "."),
          flush=True)
    entrenamientos = Entrenamientos(exp, args.minimo_libre_gb)
    d = exp.resumen["decisiones"]

    for nombre in CONFIGURACIONES:
        entrenamientos.variante(nombre, nombre, 42)
    mejor = min(CONFIGURACIONES, key=exp.logloss)
    entrenamientos.variante(f"{mejor}_s43", mejor, 43)
    ruido = abs(exp.logloss(mejor) - exp.logloss(f"{mejor}_s43"))
    d["ruido_semilla_mejor"] = ruido
    print(f"\nRuido de semilla de {mejor}: {ruido:.6f}", flush=True)

    for nombre in CONFIGURACIONES:
        if nombre != PARTIDA:
            exp.comparar(nombre, PARTIDA)
    diferencia = exp.resumen["comparaciones"].get(f"{mejor} - {PARTIDA} (todas)")
    elegida = elegir_configuracion(mejor, PARTIDA, diferencia, ruido)
    if elegida != mejor:
        entrenamientos.variante(f"{elegida}_s43", elegida, 43)
    d.update({"mejor_logloss": mejor, "elegida": elegida, "parametros_elegidos": parametros(elegida),
              "arboles_elegida": exp.resumen["variantes"][elegida]["arboles"],
              "ruido_semilla_elegida": abs(exp.logloss(elegida) - exp.logloss(f"{elegida}_s43"))})
    d["segundos_total"] = round(time.monotonic() - inicio)
    exp.guardar()

    print("\n== Validación (logloss / AUC-PR / skill) ==")
    for nombre, v in exp.resumen["variantes"].items():
        m = v["validacion"]["todas"]
        print(f"  {nombre:<8}{v['arboles']:>6} árboles  {m['logloss']:.6f}  {m['auc_pr']:.4f}  {m['brier_skill']:.4f}")
    print(f"\nMenor logloss: {mejor}. Elegida: {elegida}. Total {d['segundos_total']} s; {memoria()}")
    print(f"Guardado en {args.salida}, {args.predicciones} y {args.modelos}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
