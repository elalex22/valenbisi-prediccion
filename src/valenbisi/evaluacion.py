"""Cómo se evalúa un modelo: reparto temporal y métricas sobre probabilidades.

El reparto es por fechas, nunca aleatorio: se entrena con el pasado y se evalúa
con el futuro, como se usaría el modelo de verdad. En cada frontera se deja un
hueco de 60 min, porque la etiqueta de una fila mira a ``slot_utc + 60``: sin el
hueco, las últimas filas de un conjunto tendrían su etiqueta en el siguiente.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from valenbisi.objetivo import HORIZONTE

ZONA = "Europe/Madrid"
# Inicio de validación y de prueba, a medianoche hora local.
FRONTERAS = {
    "validacion": pd.Timestamp("2024-09-01", tz=ZONA).tz_convert("UTC"),
    "prueba": pd.Timestamp("2025-03-01", tz=ZONA).tz_convert("UTC"),
}
CONJUNTOS = ("entrenamiento", *FRONTERAS)


def repartir(df: pd.DataFrame, fronteras: dict[str, pd.Timestamp] = FRONTERAS) -> tuple[dict[str, pd.DataFrame], dict]:
    """Separa ``df`` por ``slot_utc`` en entrenamiento, validación y prueba.

    Una fila entra en un conjunto si su ``slot_utc`` y su t+60 caen los dos
    dentro de él. Las filas a menos de 60 min antes de una frontera se descartan
    y se cuentan.
    """
    inicios = [None, *fronteras.values()]
    fines = [*fronteras.values(), None]
    conjuntos, descartadas = {}, {}
    for nombre, inicio, fin in zip(CONJUNTOS, inicios, fines):
        dentro = pd.Series(True, index=df.index)
        if inicio is not None:
            dentro &= df["slot_utc"] >= inicio
        if fin is not None:
            dentro &= df["slot_utc"] < fin
            hueco = dentro & (df["slot_utc"] + HORIZONTE >= fin)
            descartadas[f"hueco_antes_de_{CONJUNTOS[CONJUNTOS.index(nombre) + 1]}"] = int(hueco.sum())
            dentro &= ~hueco
        conjuntos[nombre] = df[dentro].reset_index(drop=True)
    conteo = {"filas": {nombre: len(c) for nombre, c in conjuntos.items()}, "descartadas": descartadas}
    return conjuntos, conteo


def metricas(y: pd.Series | np.ndarray, p: pd.Series | np.ndarray, p_constante: float) -> dict:
    """AUC-PR, Brier y Brier skill score de las probabilidades ``p`` frente a la realidad ``y``.

    ``p_constante`` es la proporción de positivos de entrenamiento: el Brier de
    predecirla siempre es el suelo con el que se compara (skill score 0).
    """
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    brier = brier_score_loss(y, p)
    brier_constante = brier_score_loss(y, np.full(len(y), p_constante))
    return {
        "filas": len(y),
        "porcentaje_positivos": round(y.mean() * 100, 2),
        # Un modelo al azar saca la proporción de positivos (no 0,5 como en la AUC-ROC).
        "auc_pr": round(average_precision_score(y, p), 4),
        "brier": round(brier, 5),
        "brier_skill": round(1 - brier / brier_constante, 4),
    }


def tabla_calibracion(y: pd.Series | np.ndarray, p: pd.Series | np.ndarray, n_tramos: int = 10) -> pd.DataFrame:
    """Probabilidad media predicha frente a frecuencia real, por tramos de ``p``.

    Los tramos se cortan por cuantiles, no a anchura fija: casi todas las
    predicciones están cerca de 0 y con tramos fijos los altos quedarían vacíos.
    Los cortes repetidos se unen, así que un modelo con pocos valores distintos
    (como la persistencia) da pocos tramos.
    """
    datos = pd.DataFrame({"y": np.asarray(y, dtype=bool), "p": np.asarray(p, dtype=float)})
    cortes = np.unique(np.quantile(datos["p"], np.linspace(0, 1, n_tramos + 1)))
    if len(cortes) == 1:
        tramo = np.zeros(len(datos), dtype=int)
    else:
        tramo = pd.cut(datos["p"], cortes, include_lowest=True, labels=False)
    tabla = datos.groupby(tramo).agg(filas=("y", "size"), p_media=("p", "mean"), frecuencia_real=("y", "mean"))
    return tabla.reset_index(drop=True)
