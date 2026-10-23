"""Cómo se evalúa un modelo: reparto temporal y métricas sobre probabilidades.

El reparto es por fechas, nunca aleatorio: se entrena con el pasado y se evalúa
con el futuro, como se usaría el modelo de verdad. En cada frontera se deja un
hueco de 60 min, porque la etiqueta de una fila mira a ``slot_utc + 60``: sin el
hueco, las últimas filas de un conjunto tendrían su etiqueta en el siguiente.
"""

import pandas as pd

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
