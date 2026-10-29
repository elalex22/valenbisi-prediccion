"""Cómo se evalúa un modelo: reparto temporal y métricas sobre probabilidades.

El reparto es por fechas, nunca aleatorio: se entrena con el pasado y se evalúa
con el futuro, como se usaría el modelo de verdad. En cada frontera se deja un
hueco de 60 min, porque la etiqueta de una fila mira a ``slot_utc + 60``: sin el
hueco, las últimas filas de un conjunto tendrían su etiqueta en el siguiente.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from valenbisi.objetivo import HORIZONTE

ZONA = "Europe/Madrid"
# Inicio de validación y de prueba, a medianoche hora local.
FRONTERAS = {
    "validacion": pd.Timestamp("2024-09-01", tz=ZONA).tz_convert("UTC"),
    "prueba": pd.Timestamp("2025-03-01", tz=ZONA).tz_convert("UTC"),
}
CONJUNTOS = ("entrenamiento", *FRONTERAS)
# Reparto de la v2: la antigua prueba pasa a ser validación, y la prueba v2 son los
# datos nuevos del clon (desde el 25-09-2025, ver CLAUDE.md).
FRONTERAS_V2 = {
    "validacion": pd.Timestamp("2025-03-01", tz=ZONA).tz_convert("UTC"),
    "prueba": pd.Timestamp("2025-09-25", tz=ZONA).tz_convert("UTC"),
}


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


def limites_union(nombres: list[str], fronteras: dict[str, pd.Timestamp] = FRONTERAS) -> list[list[tuple]]:
    """Filtro de pyarrow (OR de los :func:`limites` de cada uno) que lee la unión exacta de varios conjuntos.

    Para el modelo final, que se entrena con entrenamiento + validación: las filas
    del hueco entre los dos siguen fuera, igual que cuando se leían por separado.
    """
    return [limites(nombre, fronteras) for nombre in nombres]


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
    medidas: tuple[str, ...] = ("auc_pr", "brier_skill"), decimales: int = 4,
) -> dict:
    """Intervalos de confianza de ``medidas`` (AUC-PR, Brier skill y/o logloss), con bootstrap por días.

    Las filas de un mismo día no son independientes (las mismas estaciones, el
    mismo tiempo, el mismo evento), así que se remuestrean días enteros: con filas
    sueltas los intervalos saldrían demasiado estrechos. ``diferencias`` son pares
    (a, b) de modelos de los que se quiere el intervalo de a − b, calculado en las
    mismas réplicas. La logloss es «menos es mejor»: una diferencia negativa
    significa que a mejora a b. Solo se calculan las ``medidas`` pedidas: la
    AUC-PR es la más cara (ordena todas las filas en cada réplica).
    """
    y = np.asarray(y, dtype=bool)
    calculos = {
        "auc_pr": lambda y_r, p_r, brier_constante: average_precision_score(y_r, p_r),
        "brier_skill": lambda y_r, p_r, brier_constante: 1 - brier_score_loss(y_r, p_r) / brier_constante,
        "logloss": lambda y_r, p_r, brier_constante: log_loss(y_r, p_r, labels=[False, True]),
    }
    desconocidas = set(medidas) - set(calculos)
    if desconocidas:
        raise ValueError(f"Medidas desconocidas: {sorted(desconocidas)}")

    def medir(indices: np.ndarray) -> dict[str, dict[str, float]]:
        y_r = y[indices]
        brier_constante = (brier_score_loss(y_r, np.full(len(y_r), p_constante))
                           if "brier_skill" in medidas else None)
        return {m: {k: calculos[k](y_r, p[indices], brier_constante) for k in medidas}
                for m, p in predicciones.items()}

    def con_diferencias(valores: dict) -> dict:
        for a, b in diferencias:
            valores[f"{a} - {b}"] = {k: valores[a][k] - valores[b][k] for k in medidas}
        return valores

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
            resultado[nombre][metrica] = {"estimacion": round(float(valor), decimales),
                                          "inferior": round(float(inferior), decimales),
                                          "superior": round(float(superior), decimales)}
    return resultado


def calibracion_por_mes(slot_utc: pd.Series, y: np.ndarray, predicciones: dict[str, np.ndarray]) -> pd.DataFrame:
    """Por mes local: frecuencia real, probabilidad media de cada modelo y real ÷ predicho.

    Real ÷ predicho > 1 significa que ese mes se llenó más de lo que decía el
    modelo. Sirve para ver si el límite conocido de variabilidad mensual se repite.
    """
    datos = pd.DataFrame({"mes": slot_utc.dt.tz_convert(ZONA).dt.strftime("%Y-%m").to_numpy(),
                          "y": np.asarray(y, dtype=bool), **{m: np.asarray(p, dtype=float)
                                                             for m, p in predicciones.items()}})
    tabla = datos.groupby("mes").agg(filas=("y", "size"), frecuencia_real=("y", "mean"),
                                     **{f"p_media_{m}": (m, "mean") for m in predicciones})
    for m in predicciones:
        tabla[f"real_entre_predicho_{m}"] = tabla["frecuencia_real"] / tabla[f"p_media_{m}"]
    return tabla.reset_index()


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
