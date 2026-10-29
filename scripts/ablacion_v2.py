"""Ablación de la v2: ¿qué grupos de features nuevas entran? Reglas fijadas en CLAUDE.md antes de ejecutar.

Reparto v2: entrenamiento dic-2022 a feb-2025, validación mar–sep 2025. LightGBM
con los PARAMETROS de la v1 y parada temprana en la logloss de validación. El
Dataset se construye por bloques del Parquet (no cabe entero en memoria).

Etapa 1, cada grupo por separado sobre la base (las 15 features de la v1):
  A0 base (semilla 42), A0b base (semilla 43, mide el ruido),
  A1 + meteorología con cantidad, A2 + meteorología binaria, A3 + vecinas, A4 + retardos.
Etapa 2, con los grupos aceptados: A5 todos juntos, A5b con la semilla 43 y
A5 sin cada grupo. Solo se queda un grupo si quitarlo empeora.

Reglas (logloss de validación, menos es mejor):
- Un grupo entra si mejora a A0, la mejora es más del doble del ruido de semilla
  |A0 − A0b| y el intervalo al 95 % (bootstrap por días) de la diferencia
  pareada excluye el 0.
- Retardos, además: la mejora tiene que mantenerse en las filas con los cuatro
  retardos presentes (intervalo que excluye el 0).
- Lluvia: si entran las dos versiones, gana la de menor logloss solo si su
  intervalo excluye el 0; si no, la binaria, que es más sencilla.
- Etapa 2: un grupo se queda si A5 sin él empeora con un intervalo que excluye el 0.

Además, para cada variante: medidas en todas las filas, con y sin retardos, no
llena ahora y libres 1–6; y, con los modelos con retardos, una simulación de
la primera semana de la prueba v2 (retardos vaciados en toda la validación).

Guarda las predicciones de validación (float32) en data/processed/ablacion_v2/
y data/processed/resumen_ablacion_v2.json, actualizado tras cada paso. Vigila
la RAM y evita la suspensión.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\ablacion_v2.py
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import log_loss

from valenbisi.evaluacion import FRONTERAS_V2, ZONA, intervalos_bootstrap, limites, metricas
from valenbisi.features_v2 import FEATURES_RETARDOS, FEATURES_V2
from valenbisi.memoria import comprobar_memoria, evitar_suspension, memoria_libre_gb, memoria_proceso_gb
from valenbisi.modelo import (
    FEATURES_MODELO,
    PARAMETROS,
    SecuenciaParquet,
    dataset_por_lotes,
    entrenar,
    predecir_por_lotes,
    vigilar_memoria,
)
from valenbisi.vecinas import FEATURES_VECINAS

RAIZ = Path(__file__).resolve().parents[1]
GRUPOS = {
    "meteo_cantidad": ["temp", "humedad", "viento", "nubosidad", "prec_1h", "prec_3h"],
    "meteo_binaria": ["temp", "humedad", "viento", "nubosidad", "llueve_1h"],
    "vecinas": FEATURES_VECINAS,
    "retardos": FEATURES_RETARDOS,
}
ETAPA_1 = {"A0": ([], 42), "A0b": ([], 43), "A1": (["meteo_cantidad"], 42), "A2": (["meteo_binaria"], 42),
           "A3": (["vecinas"], 42), "A4": (["retardos"], 42)}
VARIANTE_DE_GRUPO = {"meteo_cantidad": "A1", "meteo_binaria": "A2", "vecinas": "A3", "retardos": "A4"}
RETARDOS_7D = ["libres_obj_7d", "cambio_7d"]
REPLICAS = 200
MAX_ARBOLES = 3000


# ---------- Reglas (funciones puras, con tests) ----------

def acepta(diferencia: dict, ruido: float) -> bool:
    """``diferencia`` es la logloss de (variante − base): negativa = mejora."""
    e = diferencia["estimacion"]
    return e < 0 and -e > 2 * ruido and diferencia["superior"] < 0


def elegir_lluvia(acepta_cantidad: bool, acepta_binaria: bool, cantidad_menos_binaria: dict | None) -> str | None:
    if acepta_cantidad and acepta_binaria:
        return "meteo_cantidad" if cantidad_menos_binaria["superior"] < 0 else "meteo_binaria"
    if acepta_cantidad:
        return "meteo_cantidad"
    if acepta_binaria:
        return "meteo_binaria"
    return None


def se_queda(sin_grupo_menos_completo: dict) -> bool:
    """Etapa 2: quitar el grupo empeora (A5−g − A5 > 0) con un intervalo que excluye el 0."""
    return sin_grupo_menos_completo["inferior"] > 0


def columnas(grupos: list[str]) -> list[str]:
    """Base + grupos, en el orden de FEATURES_V2."""
    elegidas = set(FEATURES_MODELO) | {c for g in grupos for c in GRUPOS[g]}
    return [c for c in FEATURES_V2 if c in elegidas]


# ---------- Ejecución ----------

def memoria() -> str:
    proceso, libre = memoria_proceso_gb(), memoria_libre_gb()
    return "" if proceso is None else f"proceso {proceso[0]:.1f} GB (pico {proceso[1]:.1f}), equipo {libre:.1f} GB libres"


class Ablacion:
    def __init__(self, args):
        self.args = args
        self.filtro_ent = limites("entrenamiento", FRONTERAS_V2)
        self.filtro_val = limites("validacion", FRONTERAS_V2)
        base = SecuenciaParquet(args.features, FEATURES_MODELO, self.filtro_ent)
        self.y_ent = base.leer(["llena_60"])["llena_60"].to_numpy()
        val = SecuenciaParquet(args.features, FEATURES_MODELO, self.filtro_val)
        contexto = val.leer(["slot_utc", "llena", "libres", "llena_60", *FEATURES_RETARDOS])
        self.y_val = contexto["llena_60"].to_numpy()
        self.dias = contexto["slot_utc"].dt.tz_convert(ZONA).dt.strftime("%Y-%m-%d").to_numpy()
        con_retardos = contexto[FEATURES_RETARDOS].notna().all(axis=1).to_numpy()
        self.subgrupos = {
            "todas": np.ones(len(contexto), dtype=bool),
            "con_retardos": con_retardos,
            "sin_retardos": ~con_retardos,
            "no_llena_ahora": ~contexto["llena"].to_numpy(),
            "libres_1_6": contexto["libres"].between(1, 6).to_numpy(),
        }
        self.p_constante = float(self.y_ent.mean())
        self.predicciones: dict[str, np.ndarray] = {}
        self.resumen = {"filas": {"entrenamiento": len(self.y_ent), "validacion": len(self.y_val)},
                        "subgrupos_filas": {k: int(v.sum()) for k, v in self.subgrupos.items()},
                        "parametros": PARAMETROS, "variantes": {}, "comparaciones": {}, "decisiones": {}}
        args.predicciones.mkdir(parents=True, exist_ok=True)
        filas = f"Entrenamiento {len(self.y_ent):,} filas, validación {len(self.y_val):,}".replace(",", ".")
        print(f"{filas} ({memoria()})", flush=True)

    def guardar(self) -> None:
        self.args.salida.write_text(json.dumps(self.resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    def medir(self, p: np.ndarray) -> dict:
        salida = {}
        for nombre, filtro in self.subgrupos.items():
            if filtro.sum() == 0:
                continue
            m = metricas(self.y_val[filtro], p[filtro], self.p_constante)
            m["logloss"] = round(log_loss(self.y_val[filtro], p[filtro], labels=[False, True]), 6)
            salida[nombre] = m
        return salida

    def variante(self, nombre: str, grupos: list[str], semilla: int, simular_retardos: bool = False) -> None:
        cols = columnas(grupos)
        comprobar_memoria(self.args.minimo_libre_gb)
        print(f"\n== {nombre}: {len(cols)} features ({', '.join(grupos) or 'base'}), semilla {semilla} ==", flush=True)
        inicio = time.monotonic()
        s_ent = SecuenciaParquet(self.args.features, cols, self.filtro_ent)
        s_val = SecuenciaParquet(self.args.features, cols, self.filtro_val)
        d_ent = dataset_por_lotes(s_ent, self.y_ent)
        d_val = dataset_por_lotes(s_val, self.y_val, referencia=d_ent)
        print(f"  datos listos en {time.monotonic() - inicio:.0f} s ({memoria()})", flush=True)
        modelo = entrenar(d_ent, d_val, {**PARAMETROS, "seed": semilla}, max_arboles=MAX_ARBOLES,
                          callbacks=[vigilar_memoria(self.args.minimo_libre_gb)], registro=500)
        segundos = time.monotonic() - inicio
        del d_ent, d_val
        gc.collect()
        p = predecir_por_lotes(modelo, s_val, num_iteration=modelo.best_iteration)
        self.registrar(nombre, p, {"grupos": grupos, "features": cols, "semilla": semilla,
                                   "arboles": modelo.best_iteration, "segundos_entrenamiento": round(segundos)})
        if simular_retardos:
            for escenario, vaciar in (("sin_7d", RETARDOS_7D), ("sin_retardos", FEATURES_RETARDOS)):
                indices = [cols.index(c) for c in vaciar]
                partes = []
                for X in s_val.bloques():
                    X = X.copy()
                    X[:, indices] = np.nan
                    partes.append(modelo.predict(X, num_iteration=modelo.best_iteration,
                                                 num_threads=PARAMETROS["num_threads"]))
                self.registrar(f"{nombre}_{escenario}", np.concatenate(partes),
                               {"simulacion": escenario, "de": nombre, "vaciadas": vaciar})
        logloss = self.resumen["variantes"][nombre]["validacion"]["todas"]["logloss"]
        print(f"  {modelo.best_iteration} árboles, logloss {logloss:.6f} "
              f"({time.monotonic() - inicio:.0f} s; {memoria()})", flush=True)
        del modelo, s_ent, s_val
        gc.collect()

    def registrar(self, nombre: str, p: np.ndarray, info: dict) -> None:
        self.predicciones[nombre] = p
        np.save(self.args.predicciones / f"{nombre}.npy", p.astype("float32"))
        self.resumen["variantes"][nombre] = {**info, "validacion": self.medir(p)}
        self.guardar()

    def comparar(self, a: str, b: str, subgrupo: str = "todas") -> dict | None:
        """Intervalo de la logloss de (a − b) en ``subgrupo``, con bootstrap por días (None si no hay filas)."""
        filtro = self.subgrupos[subgrupo]
        if not filtro.any():
            self.resumen["comparaciones"][f"{a} - {b} ({subgrupo})"] = None
            return None
        pares = {a: self.predicciones[a][filtro], b: self.predicciones[b][filtro]}
        r = intervalos_bootstrap(self.y_val[filtro], pares, self.dias[filtro], self.p_constante,
                                 diferencias=[(a, b)], n=REPLICAS, medidas=("logloss",), decimales=6)
        diferencia = r[f"{a} - {b}"]["logloss"]
        self.resumen["comparaciones"][f"{a} - {b} ({subgrupo})"] = diferencia
        self.guardar()
        print(f"  {a} − {b} ({subgrupo}): {diferencia['estimacion']:+.6f} "
              f"[{diferencia['inferior']:+.6f}, {diferencia['superior']:+.6f}]", flush=True)
        return diferencia


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=RAIZ / "data" / "processed" / "valenbisi_features_v2.parquet")
    parser.add_argument("--salida", type=Path, default=RAIZ / "data" / "processed" / "resumen_ablacion_v2.json")
    parser.add_argument("--predicciones", type=Path, default=RAIZ / "data" / "processed" / "ablacion_v2")
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5,
                        help="RAM libre del equipo por debajo de la cual el script se para")
    args = parser.parse_args()
    evitar_suspension()
    inicio = time.monotonic()
    comprobar_memoria(args.minimo_libre_gb)
    maximo = pc.max(pq.read_table(args.features, columns=["slot_utc"]).column("slot_utc")).as_py()
    if maximo >= FRONTERAS_V2["prueba"]:
        raise ValueError(f"El Parquet tiene datos del periodo de la prueba v2 ({maximo})")

    a = Ablacion(args)
    d = a.resumen["decisiones"]

    # Etapa 1.
    for nombre, (grupos, semilla) in ETAPA_1.items():
        a.variante(nombre, grupos, semilla, simular_retardos="retardos" in grupos)
    ruido = abs(a.resumen["variantes"]["A0"]["validacion"]["todas"]["logloss"]
                - a.resumen["variantes"]["A0b"]["validacion"]["todas"]["logloss"])
    d["ruido_semilla"] = ruido
    print(f"\nRuido de semilla (logloss): {ruido:.6f}", flush=True)
    aceptados = {}
    for grupo, var in VARIANTE_DE_GRUPO.items():
        aceptados[grupo] = acepta(a.comparar(var, "A0"), ruido)
    if aceptados["retardos"]:
        con_retardo = a.comparar("A4", "A0", "con_retardos")
        aceptados["retardos"] = con_retardo is not None and con_retardo["superior"] < 0
        d["retardos_mejoran_con_retardo_presente"] = aceptados["retardos"]
    a.comparar("A4", "A0", "sin_retardos")
    for escenario in ("sin_7d", "sin_retardos"):
        a.comparar(f"A4_{escenario}", "A0")
    las_dos = aceptados["meteo_cantidad"] and aceptados["meteo_binaria"]
    lluvia = elegir_lluvia(aceptados["meteo_cantidad"], aceptados["meteo_binaria"],
                           a.comparar("A1", "A2") if las_dos else None)
    d["etapa_1_aceptados"] = aceptados
    d["lluvia"] = lluvia
    grupos = [g for g in (lluvia, "vecinas", "retardos") if g and aceptados.get(g)]
    d["etapa_2_candidatos"] = grupos
    a.guardar()
    print(f"\nEtapa 1: aceptados {grupos or 'ninguno'} (lluvia: {lluvia})", flush=True)

    # Etapa 2.
    if len(grupos) >= 2:
        a.variante("A5", grupos, 42, simular_retardos="retardos" in grupos)
        a.variante("A5b", grupos, 43)
        d["ruido_semilla_A5"] = abs(a.resumen["variantes"]["A5"]["validacion"]["todas"]["logloss"]
                                    - a.resumen["variantes"]["A5b"]["validacion"]["todas"]["logloss"])
        finales = []
        for g in grupos:
            a.variante(f"A5_sin_{g}", [x for x in grupos if x != g], 42)
            if se_queda(a.comparar(f"A5_sin_{g}", "A5")):
                finales.append(g)
        if "retardos" in grupos:
            for escenario in ("sin_7d", "sin_retardos"):
                a.comparar(f"A5_{escenario}", "A0")
    else:
        finales = grupos
    d["grupos_finales"] = finales
    d["features_finales"] = columnas(finales)
    d["segundos_total"] = round(time.monotonic() - inicio)
    a.guardar()

    print("\n== Validación (logloss / AUC-PR) ==")
    for nombre, v in a.resumen["variantes"].items():
        fila = "  ".join(f"{s} {m['logloss']:.6f}/{m['auc_pr']:.4f}" for s, m in v["validacion"].items())
        print(f"  {nombre:<26}{fila}")
    print(f"\nGrupos finales: {finales or 'ninguno (las features de la v1)'}; "
          f"{len(d['features_finales'])} features. Total {d['segundos_total']} s; {memoria()}")
    print(f"Guardado en {args.salida} y {args.predicciones}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
