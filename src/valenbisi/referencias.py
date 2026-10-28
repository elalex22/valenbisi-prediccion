"""Tres modelos de referencia sin machine learning, ajustados solo con entrenamiento.

- Persistencia: la probabilidad depende solo de si la estación está llena ahora.
- Frecuencia histórica: la proporción de llenas a t+60 de esa estación, a esa
  hora y en ese tipo de día (laborable o fin de semana).
- Anclajes libres: la proporción de llenas a t+60 según cuántos anclajes libres
  quedan ahora. Como ``libres == 0`` es ``llena``, contiene a la persistencia.

Todas usan solo datos de t (``llena``, ``libres``, ``slot_utc``, ``estacion``)
para predecir, y ``llena_60`` solo para ajustarse.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from valenbisi.evaluacion import ZONA

MIN_FILAS_CELDA = 30


def ajustar_persistencia(entrenamiento: pd.DataFrame) -> dict[bool, float]:
    """{llena ahora: proporción de llena_60} en entrenamiento."""
    proporcion = entrenamiento.groupby("llena")["llena_60"].mean()
    return {True: float(proporcion[True]), False: float(proporcion[False])}


def predecir_persistencia(modelo: dict[bool, float], df: pd.DataFrame) -> np.ndarray:
    return np.where(df["llena"].to_numpy(), modelo[True], modelo[False])


def calendario(df: pd.DataFrame) -> pd.DataFrame:
    """Hora local (0–23) y si es fin de semana, del instante t.

    Se usa la hora de t y no la de t+60: es la que se conoce al predecir. Los
    festivos cuentan como laborables de momento.
    """
    local = df["slot_utc"].dt.tz_convert(ZONA)
    return pd.DataFrame({"hora": local.dt.hour.to_numpy(), "fin_de_semana": (local.dt.day_of_week >= 5).to_numpy()},
                        index=df.index)


@dataclass
class FrecuenciaHistorica:
    celdas: pd.Series     # (estacion, hora, fin_de_semana) -> proporción de llena_60
    respaldo: pd.Series   # (hora, fin_de_semana) -> proporción en todas las estaciones


def ajustar_frecuencia(entrenamiento: pd.DataFrame, min_filas: int = MIN_FILAS_CELDA) -> FrecuenciaHistorica:
    """Proporción de llena_60 por celda; las celdas con menos de ``min_filas`` no se usan."""
    datos = calendario(entrenamiento).assign(estacion=entrenamiento["estacion"], llena_60=entrenamiento["llena_60"])
    por_celda = datos.groupby(["estacion", "hora", "fin_de_semana"])["llena_60"].agg(["mean", "size"])
    respaldo = datos.groupby(["hora", "fin_de_semana"])["llena_60"].mean()
    return FrecuenciaHistorica(celdas=por_celda.loc[por_celda["size"] >= min_filas, "mean"], respaldo=respaldo)


def predecir_frecuencia(modelo: FrecuenciaHistorica, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidad de cada fila y qué filas usaron el respaldo (celda sin datos suficientes)."""
    claves = calendario(df).assign(estacion=df["estacion"])
    p = claves.join(modelo.celdas.rename("p"), on=["estacion", "hora", "fin_de_semana"])["p"]
    p_respaldo = claves.join(modelo.respaldo.rename("p"), on=["hora", "fin_de_semana"])["p"]
    uso_respaldo = p.isna().to_numpy()
    return p.fillna(p_respaldo).to_numpy(dtype=float), uso_respaldo


def ajustar_libres(entrenamiento: pd.DataFrame, min_filas: int = MIN_FILAS_CELDA) -> pd.Series:
    """{libres: proporción de llena_60} en entrenamiento, solo con las celdas de al menos ``min_filas``."""
    por_valor = entrenamiento.groupby("libres")["llena_60"].agg(["mean", "size"])
    return por_valor.loc[por_valor["size"] >= min_filas, "mean"].sort_index()


def predecir_libres(modelo: pd.Series, df: pd.DataFrame) -> np.ndarray:
    """Proporción de la celda de ``libres``; sin celda válida, la de la celda válida inferior más cercana.

    Los valores sin celda son estaciones grandes con muchos huecos, donde casi
    nunca se llena: la celda inferior (ya cercana a 0) es una buena aproximación.
    Si el valor es menor que todas las celdas, se usa la más baja.
    """
    claves = modelo.index.to_numpy()
    pos = np.searchsorted(claves, df["libres"].to_numpy(), side="right") - 1
    return modelo.to_numpy(dtype=float)[np.clip(pos, 0, len(claves) - 1)]
