"""Regresión logística por tramos: la referencia sencilla de la comparación de modelos v2.

Cada variable se convierte en indicadores 0/1 en una matriz dispersa, así que el
modelo es aditivo, pero no lineal dentro de cada variable:

- numéricas (ocupación, deltas, vecinas, retardos, lat/lon…): tramos por los
  deciles de entrenamiento (cortes únicos), más un tramo «falta» para los NaN.
  No se imputa nada: la logística aprende cuánto pesa no tener el dato;
- ``libres``: valor exacto de 0 a 10, luego 11–15 y 16 o más (más «falta»);
- ``llena`` y ``festivo``: un indicador cada una;
- ``estacion``: one-hot de las estaciones de entrenamiento. Una estación nueva
  da todo ceros, así que se queda en el efecto medio;
- calendario: hora × laborable/fin de semana (48), mes (12) y día de la semana (7).

Se ajusta con SGD (log_loss, L2, promediado, paso constante) recorriendo los bloques del Parquet
en orden barajado, con parada temprana por la logloss de validación al final de
cada época (:func:`ajustar`). El modelo guardado (JSON) lleva los tramos y los
coeficientes, y predice sin scikit-learn.
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

CUANTILES = 10
# Paso constante con promedio (ASGD). El paso por defecto de scikit-learn («optimal»,
# 1/(alpha·t)) empieza con pasos enormes cuando alpha es pequeño, y el promedio los
# arrastra durante épocas: con alpha = 1e-6 la logloss se quedaba lejos de la
# solución exacta. Con 0,01 llega a ella (test frente a LogisticRegression).
ETA0 = 0.01
# Cortes de libres (searchsorted por la derecha): 0, 1, …, 10 exactos, 11–15 y 16 o más.
CORTES_LIBRES = [*range(1, 12), 16]
BINARIAS = ("llena", "festivo")
CALENDARIO = ("minuto_del_dia", "dia_semana", "mes")
ESTACION = "estacion"


def cortes_cuantiles(valores: np.ndarray, n: int = CUANTILES) -> list[float]:
    """Cortes únicos en los cuantiles 1/n … (n−1)/n de los valores no NaN (deciles con n = 10)."""
    v = valores[~np.isnan(valores)]
    if len(v) == 0:
        return []
    return [float(c) for c in np.unique(np.quantile(v.astype(np.float64), np.arange(1, n) / n))]


@dataclass
class Tramos:
    """Cómo se convierte cada columna de entrada en indicadores (se ajusta solo con entrenamiento)."""
    columnas: list[str]
    cortes: dict[str, list[float]]      # numéricas, incluida libres
    estaciones: list[int]

    def bloques(self) -> list[tuple[str, int]]:
        """(variable, número de indicadores), en el orden de las columnas de la matriz."""
        salida = []
        for c in self.columnas:
            if c in self.cortes:
                salida.append((c, len(self.cortes[c]) + 2))        # tramos + «falta»
            elif c in BINARIAS:
                salida.append((c, 1))
            elif c == ESTACION:
                salida.append((c, len(self.estaciones)))
        salida += [("hora_x_tipo_dia", 48), ("mes", 12), ("dia_semana", 7)]
        return salida

    @property
    def ancho(self) -> int:
        return sum(n for _, n in self.bloques())

    def nombres(self) -> list[str]:
        """Nombre de cada indicador, para leer los coeficientes."""
        salida = []
        for variable, n in self.bloques():
            if variable in self.cortes:
                salida += [f"{variable} tramo {i}" for i in range(n - 1)] + [f"{variable} falta"]
            elif variable in BINARIAS:
                salida.append(variable)
            elif variable == ESTACION:
                salida += [f"estacion {e}" for e in self.estaciones]
            elif variable == "hora_x_tipo_dia":
                salida += [f"hora {h} {t}" for h in range(24) for t in ("laborable", "fin de semana")]
            elif variable == "mes":
                salida += [f"mes {m}" for m in range(1, 13)]
            else:
                salida += [f"dia_semana {d}" for d in range(7)]
        return salida

    def indices(self, X: np.ndarray) -> np.ndarray:
        """(filas, variables) con la columna activa de cada variable, o −1 si no hay ninguna."""
        col = {c: X[:, j] for j, c in enumerate(self.columnas)}
        partes, desplazamiento = [], 0
        estaciones = np.asarray(self.estaciones, dtype=np.float64)
        for variable, n in self.bloques():
            if variable in self.cortes:
                x = col[variable]
                i = np.searchsorted(np.asarray(self.cortes[variable]), x, side="right")
                i[np.isnan(x)] = n - 1
            elif variable in BINARIAS:
                i = np.where(col[variable] == 1, 0, -1)
            elif variable == ESTACION:
                x = col[variable]
                if len(estaciones) == 0:
                    i = np.full(len(x), -1)
                else:
                    i = np.searchsorted(estaciones, x).clip(max=len(estaciones) - 1)
                    i = np.where(estaciones[i] == x, i, -1)
            elif variable == "hora_x_tipo_dia":
                hora = np.floor(col["minuto_del_dia"] / 60)
                dia = col["dia_semana"]
                valida = (hora >= 0) & (hora < 24) & (dia >= 0) & (dia < 7)
                i = np.where(valida, np.nan_to_num(hora) * 2 + (dia >= 5), -1)
            else:
                x = np.floor(col[variable]) - (variable == "mes")     # mes 1–12 → 0–11
                i = np.where((x >= 0) & (x < n), np.nan_to_num(x), -1)
            i = np.asarray(i, dtype=np.int64)
            partes.append(np.where(i >= 0, i + desplazamiento, -1))
            desplazamiento += n
        return np.stack(partes, axis=1).astype(np.int32)

    def matriz(self, X: np.ndarray) -> sparse.csr_matrix:
        """Matriz dispersa de indicadores (un 1 por variable, salvo estaciones nuevas o valores fuera de rango)."""
        indices = self.indices(X)
        activos = indices >= 0
        indptr = np.concatenate([[0], np.cumsum(activos.sum(axis=1))])
        columnas = indices[activos]        # por filas; dentro de cada fila, en orden creciente
        return sparse.csr_matrix((np.ones(len(columnas)), columnas, indptr), shape=(len(X), self.ancho))


def ajustar_tramos(columnas: list[str], valores: Callable[[str], np.ndarray]) -> Tramos:
    """Tramos de entrenamiento. ``valores(c)`` devuelve la columna c entera de entrenamiento (float32)."""
    cortes, estaciones = {}, []
    for c in columnas:
        if c == "libres":
            cortes[c] = [float(x) for x in CORTES_LIBRES]
        elif c == ESTACION:
            v = valores(c)
            estaciones = [int(e) for e in np.unique(v[~np.isnan(v)])]
        elif c not in BINARIAS and c not in CALENDARIO:
            cortes[c] = cortes_cuantiles(valores(c))
    faltan = [c for c in CALENDARIO if c not in columnas]
    if faltan:
        raise ValueError(f"Faltan las columnas de calendario {faltan}")
    return Tramos(list(columnas), cortes, estaciones)


@dataclass
class Logistica:
    tramos: Tramos
    coef: np.ndarray
    intercepto: float

    def predecir(self, X: np.ndarray) -> np.ndarray:
        return expit(self.tramos.matriz(X) @ self.coef + self.intercepto)

    def predecir_por_bloques(self, bloques: Iterable[np.ndarray]) -> np.ndarray:
        partes = [self.predecir(X) for X in bloques]
        return np.concatenate(partes) if partes else np.empty(0)

    def guardar(self, ruta: Path) -> None:
        # json escribe los float con todas sus cifras: el modelo cargado predice exactamente lo mismo.
        datos = {"columnas": self.tramos.columnas, "cortes": self.tramos.cortes,
                 "estaciones": self.tramos.estaciones, "nombres": self.tramos.nombres(),
                 "coef": [float(c) for c in self.coef], "intercepto": float(self.intercepto)}
        Path(ruta).write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: Path) -> "Logistica":
        d = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(Tramos(d["columnas"], d["cortes"], d["estaciones"]), np.asarray(d["coef"]), d["intercepto"])


@dataclass
class Ajuste:
    modelo: Logistica
    logloss_por_epoca: list[float]

    @property
    def mejor_epoca(self) -> int:
        return int(np.argmin(self.logloss_por_epoca)) + 1


def ajustar(tramos: Tramos, bloque: Callable[[int], tuple[np.ndarray, np.ndarray]], n_bloques: int,
            validacion: Callable[[], Iterable[np.ndarray]], y_val: np.ndarray, alpha: float, semilla: int,
            max_epocas: int = 8, paciencia: int = 1, eta0: float = ETA0,
            al_acabar_epoca: Callable[[int, float], None] | None = None) -> Ajuste:
    """SGD con pérdida logística y L2, promediado y con paso constante ``eta0``, sobre todas las filas.

    ``bloque(g)`` devuelve (X, y) del bloque g. Cada época recorre los bloques en
    un orden barajado y baraja las filas de cada bloque (si no, el SGD vería los
    datos en orden temporal); la semilla solo cambia ese barajado. Al final de
    cada época mide la logloss de validación (``validacion()`` da sus bloques en
    orden) y para tras ``paciencia`` épocas sin mejorar. Devuelve la mejor época.
    """
    sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, average=True, learning_rate="constant",
                        eta0=eta0, random_state=semilla)
    rng = np.random.default_rng(semilla)
    historial, mejor, sin_mejora = [], None, 0
    for epoca in range(1, max_epocas + 1):
        for g in rng.permutation(n_bloques):
            X, y = bloque(int(g))
            if len(y) == 0:
                continue
            orden = rng.permutation(len(y))
            sgd.partial_fit(tramos.matriz(X[orden]), np.asarray(y)[orden].astype(np.int8), classes=np.array([0, 1]))
        modelo = Logistica(tramos, sgd.coef_[0].copy(), float(sgd.intercept_[0]))
        ll = float(log_loss(y_val, modelo.predecir_por_bloques(validacion()), labels=[False, True]))
        historial.append(ll)
        if al_acabar_epoca is not None:
            al_acabar_epoca(epoca, ll)
        if mejor is None or ll < mejor[0]:
            mejor, sin_mejora = (ll, modelo), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    return Ajuste(mejor[1], historial)
