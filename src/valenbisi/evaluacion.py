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


def limites(nombre: str, fronteras: dict[str, pd.Timestamp] = FRONTERAS) -> list[tuple]:
    """Filtros de pyarrow con el ``slot_utc`` de un conjunto, con el hueco ya descontado.

    Dan las mismas filas que :func:`repartir`, pero permiten leer del Parquet solo
    un conjunto: el entrenamiento nunca carga validación ni prueba.
    """
    i = CONJUNTOS.index(nombre)
    inicio = [None, *fronteras.values()][i]
    fin = [*fronteras.values(), None][i]
    filtros = []
    if inicio is not None:
        filtros.append(("slot_utc", ">=", inicio))
    if fin is not None:
        filtros.append(("slot_utc", "<", fin - HORIZONTE))
    return filtros


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


def evaluar(conjunto: pd.DataFrame, predicciones: dict[str, np.ndarray], p_constante: float) -> dict:
    """Métricas de cada modelo en todo el conjunto y separadas por «llena ahora».

    La separación muestra lo que la cifra global esconde: la persistencia solo
    acierta porque casi todo sigue igual, y dentro de cada grupo no ordena nada.
    """
    y = conjunto["llena_60"].to_numpy()
    llena = conjunto["llena"].to_numpy()
    grupos = {"todas": np.ones(len(y), dtype=bool), "llena_ahora": llena, "no_llena_ahora": ~llena}
    return {
        grupo: {modelo: metricas(y[filtro], p[filtro], p_constante) for modelo, p in predicciones.items()}
        for grupo, filtro in grupos.items()
    }


# Tramos de anclajes libres en t, (mínimo, máximo) incluidos; None = sin máximo.
TRAMOS_LIBRES = {"0": (0, 0), "1": (1, 1), "2-3": (2, 3), "4-6": (4, 6), "7+": (7, None)}


def evaluar_por_tramos(conjunto: pd.DataFrame, predicciones: dict[str, np.ndarray], p_constante: float,
                       tramos: dict[str, tuple[int, int | None]] = TRAMOS_LIBRES) -> dict:
    """Métricas de cada modelo por tramos de ``libres`` en t.

    Separa la zona donde de verdad se decide si una estación se llena (pocos
    huecos) de la mayoría de filas, con muchos huecos, donde acertar es fácil.
    """
    y = conjunto["llena_60"].to_numpy()
    libres = conjunto["libres"].to_numpy()
    resultado = {}
    for nombre, (minimo, maximo) in tramos.items():
        filtro = (libres >= minimo) & (libres <= (np.inf if maximo is None else maximo))
        resultado[nombre] = {modelo: metricas(y[filtro], p[filtro], p_constante) for modelo, p in predicciones.items()}
    return resultado


def remuestreo_por_dias(dias: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Índices de una réplica bootstrap que remuestrea días enteros con reemplazo.

    Se sortean tantos días como hay, y cada día sorteado aporta todas sus filas.
    """
    codigos, unicos = pd.factorize(dias)
    orden = np.argsort(codigos, kind="stable")
    fines = np.cumsum(np.bincount(codigos, minlength=len(unicos)))
    inicios = fines - np.bincount(codigos, minlength=len(unicos))
    elegidos = rng.integers(0, len(unicos), len(unicos))
    return np.concatenate([orden[inicios[d]:fines[d]] for d in elegidos])


def intervalos_bootstrap(
    y: np.ndarray, predicciones: dict[str, np.ndarray], dias: np.ndarray, p_constante: float,
    diferencias: list[tuple[str, str]] = (), n: int = 200, semilla: int = 0, nivel: float = 0.95,
) -> dict:
    """Intervalos de confianza de la AUC-PR y el Brier skill, con bootstrap por días.

    Las filas de un mismo día no son independientes (las mismas estaciones, el
    mismo tiempo, el mismo evento), así que se remuestrean días enteros: con filas
    sueltas los intervalos saldrían demasiado estrechos. ``diferencias`` son pares
    (a, b) de modelos de los que se quiere el intervalo de a − b, calculado en las
    mismas réplicas.
    """
    y = np.asarray(y, dtype=bool)

    def medir(indices: np.ndarray) -> dict[str, dict[str, float]]:
        y_r = y[indices]
        brier_constante = brier_score_loss(y_r, np.full(len(y_r), p_constante))
        return {m: {"auc_pr": average_precision_score(y_r, p[indices]),
                    "brier_skill": 1 - brier_score_loss(y_r, p[indices]) / brier_constante}
                for m, p in predicciones.items()}

    def con_diferencias(medidas: dict) -> dict:
        for a, b in diferencias:
            medidas[f"{a} - {b}"] = {k: medidas[a][k] - medidas[b][k] for k in ("auc_pr", "brier_skill")}
        return medidas

    estimacion = con_diferencias(medir(np.arange(len(y))))
    rng = np.random.default_rng(semilla)
    replicas = [con_diferencias(medir(remuestreo_por_dias(dias, rng))) for _ in range(n)]

    cola = (1 - nivel) / 2
    resultado = {}
    for nombre, valores in estimacion.items():
        resultado[nombre] = {}
        for metrica, valor in valores.items():
            muestra = np.array([r[nombre][metrica] for r in replicas])
            inferior, superior = np.quantile(muestra, [cola, 1 - cola])
            resultado[nombre][metrica] = {"estimacion": round(float(valor), 4),
                                          "inferior": round(float(inferior), 4), "superior": round(float(superior), 4)}
    return resultado


def tabla_calibracion(y: pd.Series | np.ndarray, p: pd.Series | np.ndarray, n_tramos: int = 10) -> pd.DataFrame:
    """Probabilidad media predicha frente a frecuencia real, por tramos de ``p``.

    Los tramos se cortan por cuantiles, no a anchura fija: casi todas las
    predicciones están cerca de 0 y con tramos fijos los altos quedarían vacíos.
    Los cortes repetidos se unen, así que un modelo con pocos valores distintos
    (como la persistencia) da pocos tramos.
    """
    datos = pd.DataFrame({"y": np.asarray(y, dtype=bool), "p": np.asarray(p, dtype=float)})
    # Solo los cortes interiores: un valor igual a un corte va al tramo de la izquierda.
    cortes = np.unique(np.quantile(datos["p"], np.linspace(0, 1, n_tramos + 1)[1:-1]))
    tramo = np.searchsorted(cortes, datos["p"], side="left")
    tabla = datos.groupby(tramo).agg(filas=("y", "size"), p_media=("p", "mean"), frecuencia_real=("y", "mean"))
    return tabla.reset_index(drop=True)
