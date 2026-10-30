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

Las simulaciones se llaman «{variante}_sim_{escenario}» (A4_sim_sin_7d…), para
no chocar con las variantes de la etapa 2 («A5_sin_{grupo}»). Registrar dos
veces el mismo nombre es un error.

Guarda las predicciones de validación (float32) en data/processed/ablacion_v2/,
los modelos en models/ablacion_v2/ y data/processed/resumen_ablacion_v2.json,
actualizado tras cada paso. Vigila la RAM y evita la suspensión.

Con --reanudar, reutiliza del resumen anterior los entrenamientos idénticos
(mismas columnas en el mismo orden, semilla, parámetros y parada temprana) en
vez de repetirlos; el resumen dice cuáles («reutilizadas»). Si faltan las
simulaciones pero está el modelo guardado, y reproduce sus predicciones, se
simulan con él sin reentrenar. Las comparaciones se recalculan siempre.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\ablacion_v2.py [--reanudar]
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
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
    PACIENCIA,
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
ESCENARIOS_SIMULACION = {"sin_7d": RETARDOS_7D, "sin_retardos": FEATURES_RETARDOS}
TOLERANCIA_MODELO = 1e-6   # un modelo guardado vale si reproduce sus predicciones (guardadas en float32)
REPLICAS = 200
MAX_ARBOLES = 3000


# ---------- Reglas (funciones puras, con tests) ----------

def acepta(diferencia: dict, ruido: float) -> bool:
    """``diferencia`` es la logloss de (variante − base): negativa = mejora."""
    e = diferencia["estimacion"]
    return e < 0 and -e > 2 * ruido and diferencia["superior"] < 0


def retardos_se_mantienen(con_retardos_menos_base: dict | None) -> bool:
    """La mejora de A4 se mantiene en las filas con los cuatro retardos (intervalo por debajo del 0)."""
    return con_retardos_menos_base is not None and con_retardos_menos_base["superior"] < 0


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


def nombre_simulacion(variante: str, escenario: str) -> str:
    return f"{variante}_sim_{escenario}"


def columnas(grupos: list[str]) -> list[str]:
    """Base + grupos, en el orden de FEATURES_V2."""
    elegidas = set(FEATURES_MODELO) | {c for g in grupos for c in GRUPOS[g]}
    return [c for c in FEATURES_V2 if c in elegidas]


# ---------- Reanudar (funciones puras, con tests) ----------

# El primer resumen (commit 96718a0) registra los PARAMETROS de LightGBM, pero no
# el tope de árboles ni la paciencia de la parada temprana, que estaban fijos en
# el código de ese commit. Para sus variantes se toman de ahí, y se indica.
PARADA_SIN_REGISTRAR = {"max_arboles": 3000, "paciencia": 100}


def clave(features: list[str], semilla: int, parametros: dict, max_arboles: int, paciencia: int) -> str:
    """Identifica un entrenamiento: columnas (en orden), semilla, parámetros y parada temprana."""
    return json.dumps([list(features), semilla, parametros, max_arboles, paciencia], sort_keys=True)


def indice_previas(resumen: dict) -> dict[str, tuple[str, bool]]:
    """{clave: (variante, parada_asumida)} de las variantes entrenadas de un resumen anterior.

    Las simulaciones no se indexan: se reutilizan junto con su modelo.
    """
    indice = {}
    for nombre, v in resumen.get("variantes", {}).items():
        if "features" not in v:
            continue
        asumida = "max_arboles" not in v or "paciencia" not in v
        k = clave(v["features"], v["semilla"], v.get("parametros", resumen["parametros"]),
                  v.get("max_arboles", PARADA_SIN_REGISTRAR["max_arboles"]),
                  v.get("paciencia", PARADA_SIN_REGISTRAR["paciencia"]))
        indice.setdefault(k, (nombre, asumida))
    return indice


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
        # Con --reanudar, el resumen anterior se lee antes de sobrescribirlo.
        self.previo = (json.loads(args.salida.read_text(encoding="utf-8"))
                       if args.reanudar and args.salida.exists() else {})
        self.previas = indice_previas(self.previo) if self.previo else {}
        self.resumen = {"filas": {"entrenamiento": len(self.y_ent), "validacion": len(self.y_val)},
                        "subgrupos_filas": {k: int(v.sum()) for k, v in self.subgrupos.items()},
                        "parametros": PARAMETROS, "variantes": {}, "reutilizadas": {}, "comparaciones": {},
                        "decisiones": {}}
        args.predicciones.mkdir(parents=True, exist_ok=True)
        if args.reanudar:
            print(f"Reanudando: {len(self.previas)} entrenamientos anteriores reutilizables", flush=True)
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

    def reutilizar(self, nombre: str, cols: list[str], semilla: int, simular_retardos: bool) -> bool:
        """Carga un entrenamiento anterior idéntico (columnas en orden, semilla, parámetros y parada).

        Solo si están sus predicciones y, si hacen falta, las de sus simulaciones.
        Las simulaciones que falten se calculan con el modelo guardado, si lo hay
        y reproduce las predicciones.
        """
        previa = self.previas.get(clave(cols, semilla, PARAMETROS, MAX_ARBOLES, PACIENCIA))
        if previa is None:
            return False
        origen, asumida = previa
        p = self.cargar(origen)
        if p is None:
            return False
        simulaciones = {}
        if simular_retardos:
            simulaciones = {e: self.cargar(nombre_simulacion(origen, e)) for e in ESCENARIOS_SIMULACION}
            if any(s is None for s in simulaciones.values()):
                modelo = self.modelo_guardado(origen, cols, p)
                if modelo is None:
                    return False
                s_val = SecuenciaParquet(self.args.features, cols, self.filtro_val)
                for escenario in simulaciones:
                    if simulaciones[escenario] is None:
                        simulaciones[escenario] = self.simular(modelo, s_val, cols, escenario)
                        print(f"  simulación {escenario} recalculada con el modelo guardado de {origen}", flush=True)
        nota = {"reutilizada_de": origen, "parada_asumida": asumida}
        anterior = {k: v for k, v in self.previo["variantes"][origen].items() if k not in ("validacion", *nota)}
        self.registrar(nombre, p, {**anterior, **nota})
        for escenario, s in simulaciones.items():
            self.registrar(nombre_simulacion(nombre, escenario), s,
                           {"simulacion": escenario, "de": nombre, "vaciadas": ESCENARIOS_SIMULACION[escenario], **nota})
        self.resumen["reutilizadas"][nombre] = nota
        self.guardar()
        logloss = self.resumen["variantes"][nombre]["validacion"]["todas"]["logloss"]
        print(f"\n== {nombre}: reutilizada de {origen}"
              f"{' (parada sin registrar: 3000 árboles y paciencia 100 del commit 96718a0)' if asumida else ''}; "
              f"logloss {logloss:.6f} ==", flush=True)
        return True

    def cargar(self, nombre: str) -> np.ndarray | None:
        """Predicciones de una variante del resumen anterior, si están completas."""
        ruta = self.args.predicciones / f"{nombre}.npy"
        if nombre not in self.previo.get("variantes", {}) or not ruta.exists():
            return None
        p = np.load(ruta).astype("float64")
        return p if len(p) == len(self.y_val) else None

    def modelo_guardado(self, nombre: str, cols: list[str], p: np.ndarray) -> lgb.Booster | None:
        """El modelo guardado de ``nombre``, solo si tiene las mismas columnas y reproduce ``p``."""
        ruta = self.args.modelos / f"{nombre}.txt"
        if not ruta.exists():
            return None
        modelo = lgb.Booster(model_file=str(ruta))
        if modelo.feature_name() != cols:
            return None
        s_val = SecuenciaParquet(self.args.features, cols, self.filtro_val)
        diferencia = float(np.abs(predecir_por_lotes(modelo, s_val) - p).max())
        if diferencia > TOLERANCIA_MODELO:
            print(f"  {ruta.name} no reproduce sus predicciones (diferencia {diferencia:.2e}): se reentrena", flush=True)
            return None
        return modelo

    def simular(self, modelo: lgb.Booster, s_val: SecuenciaParquet, cols: list[str], escenario: str) -> np.ndarray:
        """Predicciones de validación con las columnas del escenario vaciadas (NaN)."""
        indices = [cols.index(c) for c in ESCENARIOS_SIMULACION[escenario]]
        partes = []
        for X in s_val.bloques():
            X = X.copy()
            X[:, indices] = np.nan
            partes.append(modelo.predict(X, num_threads=PARAMETROS["num_threads"]))
        return np.concatenate(partes)

    def variante(self, nombre: str, grupos: list[str], semilla: int, simular_retardos: bool = False) -> None:
        cols = columnas(grupos)
        if self.reutilizar(nombre, cols, semilla, simular_retardos):
            return
        comprobar_memoria(self.args.minimo_libre_gb)
        print(f"\n== {nombre}: {len(cols)} features ({', '.join(grupos) or 'base'}), semilla {semilla} ==", flush=True)
        inicio = time.monotonic()
        s_ent = SecuenciaParquet(self.args.features, cols, self.filtro_ent)
        s_val = SecuenciaParquet(self.args.features, cols, self.filtro_val)
        d_ent = dataset_por_lotes(s_ent, self.y_ent)
        d_val = dataset_por_lotes(s_val, self.y_val, referencia=d_ent)
        print(f"  datos listos en {time.monotonic() - inicio:.0f} s ({memoria()})", flush=True)
        modelo = entrenar(d_ent, d_val, {**PARAMETROS, "seed": semilla}, max_arboles=MAX_ARBOLES,
                          paciencia=PACIENCIA, callbacks=[vigilar_memoria(self.args.minimo_libre_gb)], registro=500)
        segundos = time.monotonic() - inicio
        del d_ent, d_val
        gc.collect()
        # Se guarda solo hasta el mejor árbol, y se predice con el modelo ya guardado:
        # así lo que se reutilice después es exactamente lo que se midió.
        self.args.modelos.mkdir(parents=True, exist_ok=True)
        ruta_modelo = self.args.modelos / f"{nombre}.txt"
        modelo.save_model(str(ruta_modelo), num_iteration=modelo.best_iteration)
        modelo = lgb.Booster(model_file=str(ruta_modelo))
        p = predecir_por_lotes(modelo, s_val)
        self.registrar(nombre, p, {"grupos": grupos, "features": cols, "semilla": semilla, "parametros": PARAMETROS,
                                   "max_arboles": MAX_ARBOLES, "paciencia": PACIENCIA,
                                   "arboles": modelo.num_trees(), "segundos_entrenamiento": round(segundos)})
        if simular_retardos:
            for escenario, vaciar in ESCENARIOS_SIMULACION.items():
                self.registrar(nombre_simulacion(nombre, escenario), self.simular(modelo, s_val, cols, escenario),
                               {"simulacion": escenario, "de": nombre, "vaciadas": vaciar})
        logloss = self.resumen["variantes"][nombre]["validacion"]["todas"]["logloss"]
        print(f"  {modelo.num_trees()} árboles, logloss {logloss:.6f} "
              f"({time.monotonic() - inicio:.0f} s; {memoria()})", flush=True)
        del modelo, s_ent, s_val
        gc.collect()

    def registrar(self, nombre: str, p: np.ndarray, info: dict) -> None:
        if nombre in self.resumen["variantes"]:
            raise ValueError(f"El nombre {nombre!r} ya está registrado: se sobrescribirían sus predicciones")
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
    parser.add_argument("--modelos", type=Path, default=RAIZ / "models" / "ablacion_v2")
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5,
                        help="RAM libre del equipo por debajo de la cual el script se para")
    parser.add_argument("--reanudar", action="store_true",
                        help="reutiliza los entrenamientos del resumen anterior idénticos en columnas (y su orden), "
                             "semilla, parámetros y parada temprana")
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
        aceptados["retardos"] = retardos_se_mantienen(a.comparar("A4", "A0", "con_retardos"))
        d["retardos_mejoran_con_retardo_presente"] = aceptados["retardos"]
    a.comparar("A4", "A0", "sin_retardos")
    for escenario in ESCENARIOS_SIMULACION:
        a.comparar(nombre_simulacion("A4", escenario), "A0")
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
            for escenario in ESCENARIOS_SIMULACION:
                a.comparar(nombre_simulacion("A5", escenario), "A0")
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
    print(f"Guardado en {args.salida}, {args.predicciones} y {args.modelos}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
