"""Búsqueda de hiperparámetros de LightGBM con las 23 features de la v2. Reglas fijadas en CLAUDE.md antes de ejecutar.

Reparto v2: entrenamiento dic-2022 a feb-2025, validación mar–sep 2025. Seis
configuraciones (L0 = los PARAMETROS de la v1, que es A5 de la ablación y se
reutiliza), todas con parada temprana por la logloss de validación:

  L0 63 hojas / 1000 filas por hoja / L2 1     L1 31 / 1000 / 1
  L2 127 / 1000 / 1                           L3 255 / 1000 / 1
  L4 127 / 5000 / 10                          L5 255 / 5000 / 10

Regla: la de menor logloss sustituye a L0 solo si mejora más del doble del
ruido de semilla de L0 (A5 frente a A5b) y el intervalo al 95 % (bootstrap
por días) de la diferencia pareada excluye el 0. Si gana otra, se repite con
la semilla 43 para medir su ruido.

El Dataset se construye una vez, por bloques del Parquet, y sirve para todas
las configuraciones (los cortes no dependen de ellas). Guarda las
predicciones de validación en data/processed/modelos_v2/, los modelos en
models/modelos_v2/ y data/processed/resumen_lightgbm_v2.json, actualizado
tras cada paso. Vigila la RAM y evita la suspensión.

Con --reanudar, reutiliza los entrenamientos idénticos (columnas en el mismo
orden, semilla, parámetros y parada temprana) del resumen anterior. Los de la
ablación v2 (A5 y A5b) se reutilizan siempre.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\hiperparametros_lightgbm_v2.py [--reanudar]
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import lightgbm as lgb

from valenbisi.experimento import Experimento, Fuente
from valenbisi.memoria import comprobar_memoria, evitar_suspension, memoria_libre_gb, memoria_proceso_gb
from valenbisi.modelo import PACIENCIA, PARAMETROS, dataset_por_lotes, entrenar, predecir_por_lotes, vigilar_memoria
from valenbisi.reanudar import clave
from valenbisi.seleccion import elegir_configuracion

RAIZ = Path(__file__).resolve().parents[1]
PROCESADOS = RAIZ / "data" / "processed"
CONFIGURACIONES = {
    "L0": {},
    "L1": {"num_leaves": 31},
    "L2": {"num_leaves": 127},
    "L3": {"num_leaves": 255},
    "L4": {"num_leaves": 127, "min_data_in_leaf": 5000, "lambda_l2": 10.0},
    "L5": {"num_leaves": 255, "min_data_in_leaf": 5000, "lambda_l2": 10.0},
}
PARTIDA = "L0"
MAX_ARBOLES = 3000
REPLICAS = 200


def parametros(nombre: str) -> dict:
    """Los PARAMETROS de la v1 con los cambios de la configuración (L0 = PARAMETROS tal cual)."""
    return {**PARAMETROS, **CONFIGURACIONES[nombre]}


def memoria() -> str:
    proceso, libre = memoria_proceso_gb(), memoria_libre_gb()
    return "" if proceso is None else f"proceso {proceso[0]:.1f} GB (pico {proceso[1]:.1f}), equipo {libre:.1f} GB libres"


class Busqueda:
    def __init__(self, exp: Experimento, minimo_libre_gb: float):
        self.exp = exp
        self.minimo_libre_gb = minimo_libre_gb
        self.datos: tuple[lgb.Dataset, lgb.Dataset] | None = None

    def preparar(self) -> tuple[lgb.Dataset, lgb.Dataset]:
        if self.datos is None:
            inicio = time.monotonic()
            d_ent = dataset_por_lotes(self.exp.secuencia("entrenamiento"), self.exp.y_ent)
            d_val = dataset_por_lotes(self.exp.secuencia("validacion"), self.exp.y_val, referencia=d_ent)
            self.datos = (d_ent, d_val)
            print(f"  datos listos en {time.monotonic() - inicio:.0f} s ({memoria()})", flush=True)
        return self.datos

    def variante(self, nombre: str, config: str, semilla: int) -> None:
        params = parametros(config)
        if self.exp.reutilizar(nombre, clave(self.exp.columnas, semilla, params, MAX_ARBOLES, PACIENCIA)):
            return
        comprobar_memoria(self.minimo_libre_gb)
        print(f"\n== {nombre}: {CONFIGURACIONES[config] or 'PARAMETROS de la v1'}, semilla {semilla} ==", flush=True)
        d_ent, d_val = self.preparar()
        inicio = time.monotonic()
        modelo = entrenar(d_ent, d_val, {**params, "seed": semilla}, max_arboles=MAX_ARBOLES, paciencia=PACIENCIA,
                          callbacks=[vigilar_memoria(self.minimo_libre_gb)], registro=500)
        segundos = time.monotonic() - inicio
        tope = modelo.current_iteration() >= MAX_ARBOLES
        # Se guarda hasta el mejor árbol y se predice con el modelo guardado: lo que se reutilice es lo medido.
        ruta = self.exp.modelos / f"{nombre}.txt"
        modelo.save_model(str(ruta), num_iteration=modelo.best_iteration)
        del modelo
        modelo = lgb.Booster(model_file=str(ruta))
        p = predecir_por_lotes(modelo, self.exp.secuencia("validacion"))
        self.exp.registrar(nombre, p, {"configuracion": config, "features": self.exp.columnas, "semilla": semilla,
                                       "parametros": params, "max_arboles": MAX_ARBOLES, "paciencia": PACIENCIA,
                                       "arboles": modelo.num_trees(), "tope_alcanzado": tope,
                                       "segundos_entrenamiento": round(segundos), "modelo": ruta.name})
        print(f"  {modelo.num_trees()} árboles{' (TOPE)' if tope else ''}, logloss {self.exp.logloss(nombre):.6f} "
              f"({segundos:.0f} s; {memoria()})", flush=True)
        del modelo
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=PROCESADOS / "valenbisi_features_v2.parquet")
    parser.add_argument("--salida", type=Path, default=PROCESADOS / "resumen_lightgbm_v2.json")
    parser.add_argument("--predicciones", type=Path, default=PROCESADOS / "modelos_v2")
    parser.add_argument("--modelos", type=Path, default=RAIZ / "models" / "modelos_v2")
    parser.add_argument("--ablacion", type=Path, default=PROCESADOS / "resumen_ablacion_v2.json",
                        help="resumen de la ablación v2, de donde se reutilizan A5 y A5b")
    parser.add_argument("--ablacion-predicciones", type=Path, default=PROCESADOS / "ablacion_v2")
    parser.add_argument("--ablacion-modelos", type=Path, default=RAIZ / "models" / "ablacion_v2")
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5,
                        help="RAM libre del equipo por debajo de la cual el script se para")
    parser.add_argument("--reanudar", action="store_true",
                        help="reutiliza los entrenamientos idénticos del resumen anterior")
    args = parser.parse_args()
    evitar_suspension()
    inicio = time.monotonic()
    comprobar_memoria(args.minimo_libre_gb)

    exp = Experimento(args.features, args.salida, args.predicciones, args.modelos, reanudar=args.reanudar,
                      fuentes=[Fuente(args.ablacion, args.ablacion_predicciones, args.ablacion_modelos)],
                      replicas=REPLICAS)
    print(f"Entrenamiento {len(exp.y_ent):,} filas, validación {len(exp.y_val):,} ({memoria()})".replace(",", "."),
          flush=True)
    busqueda = Busqueda(exp, args.minimo_libre_gb)
    d = exp.resumen["decisiones"]

    for nombre in CONFIGURACIONES:
        busqueda.variante(nombre, nombre, 42)
    busqueda.variante(f"{PARTIDA}_s43", PARTIDA, 43)
    ruido = abs(exp.logloss(PARTIDA) - exp.logloss(f"{PARTIDA}_s43"))
    d["ruido_semilla_partida"] = ruido
    print(f"\nRuido de semilla de {PARTIDA}: {ruido:.6f}", flush=True)

    for nombre in CONFIGURACIONES:
        if nombre != PARTIDA:
            exp.comparar(nombre, PARTIDA)
    mejor = min(CONFIGURACIONES, key=exp.logloss)
    diferencia = exp.resumen["comparaciones"].get(f"{mejor} - {PARTIDA} (todas)")
    elegida = elegir_configuracion(mejor, PARTIDA, diferencia, ruido)
    d.update({"mejor_logloss": mejor, "elegida": elegida, "parametros_elegidos": parametros(elegida),
              "arboles_elegida": exp.resumen["variantes"][elegida]["arboles"]})
    if elegida != PARTIDA:
        busqueda.variante(f"{elegida}_s43", elegida, 43)
        d["ruido_semilla_elegida"] = abs(exp.logloss(elegida) - exp.logloss(f"{elegida}_s43"))
    else:
        d["ruido_semilla_elegida"] = ruido
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
