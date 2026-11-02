"""Infraestructura común de los scripts que comparan modelos en la validación v2.

:class:`Experimento` lee las etiquetas y el contexto de validación una vez,
mide cada variante (todas las filas y subgrupos), guarda sus predicciones
(float32) y el resumen tras cada paso, compara variantes con bootstrap por días
y reutiliza entrenamientos idénticos de resúmenes anteriores (``--reanudar``, o
la ablación v2 como fuente fija).
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import log_loss

from valenbisi.evaluacion import FRONTERAS_V2, ZONA, intervalos_bootstrap, limites, metricas
from valenbisi.features_v2 import FEATURES_RETARDOS, FEATURES_V2_FINALES
from valenbisi.modelo import SecuenciaParquet
from valenbisi.reanudar import comprobar_nombre_nuevo, indice_previas


@dataclass(frozen=True)
class Fuente:
    """Un resumen anterior con sus carpetas de predicciones y modelos."""
    resumen: Path
    predicciones: Path
    modelos: Path


def codigos_de_dia(slot_utc: pd.Series) -> np.ndarray:
    """Un entero por día local (Europe/Madrid), numerados por orden de aparición."""
    return pd.factorize(slot_utc.dt.tz_convert(ZONA).dt.normalize())[0].astype(np.int32)


def comprobar_sin_prueba_v2(ruta: Path) -> None:
    maximo = pc.max(pq.read_table(ruta, columns=["slot_utc"]).column("slot_utc")).as_py()
    if maximo >= FRONTERAS_V2["prueba"]:
        raise ValueError(f"El Parquet tiene datos del periodo de la prueba v2 ({maximo})")


class Experimento:
    def __init__(self, features: Path, salida: Path, predicciones: Path, modelos: Path, reanudar: bool = False,
                 fuentes: list[Fuente] = (), replicas: int = 200, columnas: list[str] = FEATURES_V2_FINALES):
        comprobar_sin_prueba_v2(features)
        self.features, self.salida, self.carpeta, self.modelos = features, salida, predicciones, modelos
        self.columnas = list(columnas)
        self.replicas = replicas
        self.filtro_ent = limites("entrenamiento", FRONTERAS_V2)
        self.filtro_val = limites("validacion", FRONTERAS_V2)
        self.y_ent = self.secuencia("entrenamiento").leer(["llena_60"])["llena_60"].to_numpy()
        contexto = self.secuencia("validacion").leer(["slot_utc", "llena", "libres", "llena_60", *FEATURES_RETARDOS])
        self.y_val = contexto["llena_60"].to_numpy()
        # El día local de cada fila como código entero (orden de aparición), no como texto: el bootstrap
        # da exactamente lo mismo y ocupa 0,02 GB en vez de 0,3 GB.
        self.dias = codigos_de_dia(contexto["slot_utc"])
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
        # El resumen propio anterior (con --reanudar) se lee antes de sobrescribirlo y va primero.
        propias = [Fuente(salida, predicciones, modelos)] if reanudar and salida.exists() else []
        self.fuentes = []
        for fuente in [*propias, *fuentes]:
            if fuente.resumen.exists():
                resumen = json.loads(fuente.resumen.read_text(encoding="utf-8"))
                self.fuentes.append((fuente, resumen, indice_previas(resumen)))
        self.resumen = {"filas": {"entrenamiento": len(self.y_ent), "validacion": len(self.y_val)},
                        "subgrupos_filas": {k: int(v.sum()) for k, v in self.subgrupos.items()},
                        "columnas": self.columnas, "variantes": {}, "reutilizadas": {}, "comparaciones": {},
                        "decisiones": {}}
        predicciones.mkdir(parents=True, exist_ok=True)
        modelos.mkdir(parents=True, exist_ok=True)

    def secuencia(self, conjunto: str, columnas: list[str] | None = None) -> SecuenciaParquet:
        filtro = self.filtro_ent if conjunto == "entrenamiento" else self.filtro_val
        return SecuenciaParquet(self.features, columnas or self.columnas, filtro)

    def guardar(self) -> None:
        self.salida.write_text(json.dumps(self.resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    def medir(self, p: np.ndarray) -> dict:
        salida = {}
        for nombre, filtro in self.subgrupos.items():
            if filtro.sum() == 0:
                continue
            m = metricas(self.y_val[filtro], p[filtro], self.p_constante)
            m["logloss"] = round(log_loss(self.y_val[filtro], p[filtro], labels=[False, True]), 6)
            salida[nombre] = m
        return salida

    def logloss(self, nombre: str) -> float:
        return self.resumen["variantes"][nombre]["validacion"]["todas"]["logloss"]

    def registrar(self, nombre: str, p: np.ndarray, info: dict) -> None:
        comprobar_nombre_nuevo(nombre, self.resumen["variantes"])
        if len(p) != len(self.y_val):
            raise ValueError(f"{nombre}: {len(p)} predicciones para {len(self.y_val)} filas de validación")
        self.predicciones[nombre] = p
        np.save(self.carpeta / f"{nombre}.npy", p.astype("float32"))
        self.resumen["variantes"][nombre] = {**info, "validacion": self.medir(p)}
        self.guardar()

    def reutilizar(self, nombre: str, clave: str) -> bool:
        """Registra un entrenamiento anterior idéntico, si están sus predicciones; copia también su modelo."""
        for fuente, resumen, indice in self.fuentes:
            if clave not in indice:
                continue
            origen, _ = indice[clave]
            ruta = fuente.predicciones / f"{origen}.npy"
            if not ruta.exists():
                continue
            p = np.load(ruta).astype("float64")
            if len(p) != len(self.y_val):
                continue
            info = {k: v for k, v in resumen["variantes"][origen].items()
                    if k not in ("validacion", "reutilizada_de", "parada_asumida")}
            modelo = info.get("modelo", f"{origen}.txt")
            if (fuente.modelos / modelo).exists():
                destino = f"{nombre}{Path(modelo).suffix}"
                if (fuente.modelos / modelo).resolve() != (self.modelos / destino).resolve():
                    shutil.copyfile(fuente.modelos / modelo, self.modelos / destino)
                info["modelo"] = destino
            nota = {"resumen": fuente.resumen.name, "variante": origen}
            self.registrar(nombre, p, {**info, "reutilizada_de": nota})
            self.resumen["reutilizadas"][nombre] = nota
            self.guardar()
            print(f"\n== {nombre}: reutilizada de {fuente.resumen.name} ({origen}); "
                  f"logloss {self.logloss(nombre):.6f} ==", flush=True)
            return True
        return False

    def comparar(self, a: str, b: str, subgrupo: str = "todas") -> dict | None:
        """Intervalo de la logloss de (a − b) en ``subgrupo``, con bootstrap por días (None si no hay filas)."""
        filtro = self.subgrupos[subgrupo]
        nombre = f"{a} - {b} ({subgrupo})"
        if not filtro.any():
            self.resumen["comparaciones"][nombre] = None
            return None
        pares = {a: self.predicciones[a][filtro], b: self.predicciones[b][filtro]}
        r = intervalos_bootstrap(self.y_val[filtro], pares, self.dias[filtro], self.p_constante,
                                 diferencias=[(a, b)], n=self.replicas, medidas=("logloss",), decimales=6)
        diferencia = r[f"{a} - {b}"]["logloss"]
        self.resumen["comparaciones"][nombre] = diferencia
        self.guardar()
        print(f"  {a} − {b} ({subgrupo}): {diferencia['estimacion']:+.6f} "
              f"[{diferencia['inferior']:+.6f}, {diferencia['superior']:+.6f}]", flush=True)
        return diferencia
