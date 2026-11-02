"""Regresión logística por tramos con las 23 features de la v2: la referencia sencilla. Reglas en CLAUDE.md.

Reparto v2: entrenamiento dic-2022 a feb-2025, validación mar–sep 2025. Todas
las variables pasan a indicadores (``src/valenbisi/logistica.py``) y se ajusta
con SGD (log_loss, L2, promediado, paso constante) por bloques del Parquet,
con todas las filas, hasta 8 épocas y parada temprana por la logloss de
validación (paciencia de 1 época). Dos configuraciones:

  R1 alpha = 1e-5 (la de partida, más regularizada)     R2 alpha = 1e-6

La de menor logloss se repite con la semilla 43 (solo cambia el barajado) para
medir el ruido. Sustituye a R1 solo si mejora más del doble de ese ruido y el
intervalo al 95 % (bootstrap por días) de la diferencia pareada excluye el 0.
Si se queda R1 sin ser la de menor logloss, también se repite R1 con la semilla
43, para tener el ruido de la configuración elegida.

Los tramos (deciles de entrenamiento) se calculan una vez, leyendo columna a
columna. Guarda las predicciones de validación en data/processed/modelos_v2/,
los modelos (tramos y coeficientes, JSON) en models/modelos_v2/ y
data/processed/resumen_logistica_v2.json, actualizado tras cada paso. Vigila la
RAM y evita la suspensión.

Con --reanudar, reutiliza los ajustes idénticos (columnas en el mismo orden,
semilla y parámetros) del resumen anterior.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\logistica_v2.py [--reanudar]
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

from valenbisi.experimento import Experimento
from valenbisi.logistica import CUANTILES, ETA0, Logistica, Tramos, ajustar, ajustar_tramos
from valenbisi.memoria import comprobar_memoria, evitar_suspension, memoria_libre_gb, memoria_proceso_gb
from valenbisi.reanudar import clave
from valenbisi.seleccion import elegir_configuracion

RAIZ = Path(__file__).resolve().parents[1]
PROCESADOS = RAIZ / "data" / "processed"
CONFIGURACIONES = {"R1": {"alpha": 1e-5}, "R2": {"alpha": 1e-6}}
PARTIDA = "R1"
MAX_EPOCAS = 8
PACIENCIA_EPOCAS = 1
REPLICAS = 200


def parametros(nombre: str) -> dict:
    """Todo lo que define el ajuste (para --reanudar): no hay árboles, así que la parada va aquí."""
    return {"modelo": "logistica_tramos", "cuantiles": CUANTILES, "sgd": "log_loss, l2, average, constant",
            "eta0": ETA0, "max_epocas": MAX_EPOCAS, "paciencia_epocas": PACIENCIA_EPOCAS,
            **CONFIGURACIONES[nombre]}


def memoria() -> str:
    proceso, libre = memoria_proceso_gb(), memoria_libre_gb()
    return "" if proceso is None else f"proceso {proceso[0]:.1f} GB (pico {proceso[1]:.1f}), equipo {libre:.1f} GB libres"


class Ajustes:
    def __init__(self, exp: Experimento, minimo_libre_gb: float):
        self.exp = exp
        self.minimo_libre_gb = minimo_libre_gb
        self.tramos: Tramos | None = None

    def preparar(self) -> Tramos:
        if self.tramos is None:
            inicio = time.monotonic()

            def valores(c: str) -> np.ndarray:
                partes = list(self.exp.secuencia("entrenamiento", [c]).bloques())
                return np.concatenate(partes)[:, 0]
            self.tramos = ajustar_tramos(self.exp.columnas, valores)
            self.exp.resumen["tramos"] = {"indicadores": self.tramos.ancho,
                                          "estaciones": len(self.tramos.estaciones),
                                          "por_variable": dict(self.tramos.bloques())}
            print(f"  tramos listos en {time.monotonic() - inicio:.0f} s: {self.tramos.ancho} indicadores "
                  f"({memoria()})", flush=True)
        return self.tramos

    def variante(self, nombre: str, config: str, semilla: int) -> None:
        params = parametros(config)
        if self.exp.reutilizar(nombre, clave(self.exp.columnas, semilla, params, None, None)):
            return
        comprobar_memoria(self.minimo_libre_gb)
        print(f"\n== {nombre}: alpha {params['alpha']:g}, semilla {semilla} ==", flush=True)
        tramos = self.preparar()
        ent = self.exp.secuencia("entrenamiento")
        y_ent = self.exp.y_ent

        def bloque(g: int) -> tuple[np.ndarray, np.ndarray]:
            if len(ent.filas_bloque[g]) == 0:
                return np.empty((0, len(ent.columnas)), dtype=np.float32), y_ent[:0]
            return ent.bloque(g), y_ent[ent.inicio[g]:ent.fin[g]]

        def al_acabar_epoca(epoca: int, ll: float) -> None:
            print(f"  época {epoca}: logloss de validación {ll:.6f} ({time.monotonic() - inicio:.0f} s; "
                  f"{memoria()})", flush=True)
            comprobar_memoria(self.minimo_libre_gb)

        inicio = time.monotonic()
        ajuste = ajustar(tramos, bloque, len(ent.filas_bloque), lambda: self.exp.secuencia("validacion").bloques(),
                         self.exp.y_val, alpha=params["alpha"], semilla=semilla, max_epocas=MAX_EPOCAS,
                         paciencia=PACIENCIA_EPOCAS, al_acabar_epoca=al_acabar_epoca)
        segundos = time.monotonic() - inicio
        # Se guarda y se predice con el modelo guardado: lo que se reutilice es lo medido.
        ruta = self.exp.modelos / f"{nombre}.json"
        ajuste.modelo.guardar(ruta)
        del ajuste.modelo
        modelo = Logistica.cargar(ruta)
        p = modelo.predecir_por_bloques(self.exp.secuencia("validacion").bloques())
        self.exp.registrar(nombre, p, {"configuracion": config, "features": self.exp.columnas, "semilla": semilla,
                                       "parametros": params, "max_arboles": None, "paciencia": None,
                                       "epocas": len(ajuste.logloss_por_epoca), "mejor_epoca": ajuste.mejor_epoca,
                                       "logloss_por_epoca": [round(x, 6) for x in ajuste.logloss_por_epoca],
                                       "tope_alcanzado": ajuste.mejor_epoca == MAX_EPOCAS,
                                       "segundos_entrenamiento": round(segundos), "modelo": ruta.name})
        print(f"  mejor época {ajuste.mejor_epoca} de {len(ajuste.logloss_por_epoca)}, logloss "
              f"{self.exp.logloss(nombre):.6f} ({segundos:.0f} s; {memoria()})", flush=True)
        del modelo
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=PROCESADOS / "valenbisi_features_v2.parquet")
    parser.add_argument("--salida", type=Path, default=PROCESADOS / "resumen_logistica_v2.json")
    parser.add_argument("--predicciones", type=Path, default=PROCESADOS / "modelos_v2")
    parser.add_argument("--modelos", type=Path, default=RAIZ / "models" / "modelos_v2")
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5,
                        help="RAM libre del equipo por debajo de la cual el script se para")
    parser.add_argument("--reanudar", action="store_true",
                        help="reutiliza los ajustes idénticos del resumen anterior")
    args = parser.parse_args()
    evitar_suspension()
    inicio = time.monotonic()
    comprobar_memoria(args.minimo_libre_gb)

    exp = Experimento(args.features, args.salida, args.predicciones, args.modelos, reanudar=args.reanudar,
                      replicas=REPLICAS)
    print(f"Entrenamiento {len(exp.y_ent):,} filas, validación {len(exp.y_val):,} ({memoria()})".replace(",", "."),
          flush=True)
    ajustes = Ajustes(exp, args.minimo_libre_gb)
    d = exp.resumen["decisiones"]

    for nombre in CONFIGURACIONES:
        ajustes.variante(nombre, nombre, 42)
    mejor = min(CONFIGURACIONES, key=exp.logloss)
    ajustes.variante(f"{mejor}_s43", mejor, 43)
    ruido = abs(exp.logloss(mejor) - exp.logloss(f"{mejor}_s43"))
    d["ruido_semilla_mejor"] = ruido
    print(f"\nRuido de semilla de {mejor}: {ruido:.6f}", flush=True)

    for nombre in CONFIGURACIONES:
        if nombre != PARTIDA:
            exp.comparar(nombre, PARTIDA)
    diferencia = exp.resumen["comparaciones"].get(f"{mejor} - {PARTIDA} (todas)")
    elegida = elegir_configuracion(mejor, PARTIDA, diferencia, ruido)
    if elegida != mejor:
        ajustes.variante(f"{elegida}_s43", elegida, 43)
    d.update({"mejor_logloss": mejor, "elegida": elegida, "parametros_elegidos": parametros(elegida),
              "epocas_elegida": exp.resumen["variantes"][elegida]["mejor_epoca"],
              "ruido_semilla_elegida": abs(exp.logloss(elegida) - exp.logloss(f"{elegida}_s43"))})
    d["segundos_total"] = round(time.monotonic() - inicio)
    exp.guardar()

    print("\n== Validación (logloss / AUC-PR / skill) ==")
    for nombre, v in exp.resumen["variantes"].items():
        m = v["validacion"]["todas"]
        print(f"  {nombre:<8} época {v['mejor_epoca']}/{v['epocas']}  {m['logloss']:.6f}  {m['auc_pr']:.4f}  "
              f"{m['brier_skill']:.4f}")
    print(f"\nMenor logloss: {mejor}. Elegida: {elegida}. Total {d['segundos_total']} s; {memoria()}")
    print(f"Guardado en {args.salida}, {args.predicciones} y {args.modelos}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
