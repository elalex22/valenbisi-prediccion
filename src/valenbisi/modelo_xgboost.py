"""XGBoost por bloques del Parquet, como alternativa a LightGBM en la comparación de modelos v2.

- **Memoria**: el entrenamiento usa un ``QuantileDMatrix`` construido desde un
  :class:`IteradorParquet`, que pasa los bloques (row groups) uno a uno. Guarda
  los datos ya cuantizados y nunca la matriz float32 entera. Medido con datos
  reales: unos 0,14 GB por millón de filas (≈ 6 bytes por valor, no 1, porque
  con NaN la matriz no es densa y cada valor guarda su índice de bin en 4
  bytes), más unos 0,1 GB fijos.
- Validación se cuantiza con los mismos cortes (``ref``). Con ``cache`` va a
  disco (``ExtMemQuantileDMatrix``): solo se lee para evaluar cada árbol, así
  que cuesta poco tiempo, y da exactamente el mismo modelo (test). En disco,
  el entrenamiento sería unas 18 veces más lento, así que se queda en memoria.
- **NaN**: nativos. En cada corte, XGBoost aprende a qué lado van los que faltan.
- **Estación**: categórica nativa (``feature_types`` ``"c"``), con particiones
  por conjuntos de estaciones. El número de estación es el código de categoría,
  y cada partición reparte todos los códigos de 0 al máximo visto, también los
  que no tienen filas. Una estación nueva con un número dentro de ese rango cae
  en un lado que no se aprendió de los datos; con un número mayor, en la rama
  por defecto. Las estaciones nuevas de la prueba v2 se miden aparte.
- Sin pesos de clase: pérdida logística tal cual, por la calibración.
"""

from pathlib import Path

import numpy as np
import xgboost as xgb

from valenbisi.features import CATEGORICAS
from valenbisi.memoria import comprobar_memoria
from valenbisi.modelo import SecuenciaParquet

PARAMETROS_XGB = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_bin": 256,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    # Suma de hessianos p(1−p) ≈ 0,055 por fila con un 6 % de positivos: 50 ≈ 1000 filas,
    # como min_data_in_leaf de LightGBM.
    "min_child_weight": 50,
    "nthread": 12,
    "verbosity": 1,
}


def tipos(columnas: list[str]) -> list[str]:
    """``"c"`` para las categóricas (la estación) y ``"q"`` para las demás."""
    return ["c" if c in CATEGORICAS else "q" for c in columnas]


class IteradorParquet(xgb.DataIter):
    """Los bloques de una :class:`SecuenciaParquet` con sus etiquetas, para un ``QuantileDMatrix``."""

    def __init__(self, secuencia: SecuenciaParquet, y: np.ndarray, cache: Path | None = None):
        self.secuencia = secuencia
        self.y = np.asarray(y, dtype=np.float32)
        self.grupos = [g for g, filas in enumerate(secuencia.filas_bloque) if len(filas)]
        self.i = 0
        super().__init__(cache_prefix=None if cache is None else str(cache), release_data=True)

    def next(self, input_data) -> bool:
        if self.i == len(self.grupos):
            return False
        g = self.grupos[self.i]
        s = self.secuencia
        input_data(data=s.bloque(g), label=self.y[s.inicio[g]:s.fin[g]], feature_names=s.columnas,
                   feature_types=tipos(s.columnas))
        self.i += 1
        return True

    def reset(self) -> None:
        self.i = 0


def matrices(entrenamiento: SecuenciaParquet, y_ent: np.ndarray, validacion: SecuenciaParquet,
             y_val: np.ndarray, max_bin: int = PARAMETROS_XGB["max_bin"],
             cache: Path | None = None) -> tuple[xgb.DMatrix, xgb.DMatrix]:
    """Matrices de entrenamiento y de validación (con los cortes de entrenamiento).

    Con ``cache`` (una carpeta), la de validación va a disco ahí. La carpeta se
    puede borrar cuando ya no se usen las matrices.
    """
    d_ent = xgb.QuantileDMatrix(IteradorParquet(entrenamiento, y_ent), enable_categorical=True, max_bin=max_bin)
    if cache is None:
        d_val = xgb.QuantileDMatrix(IteradorParquet(validacion, y_val), enable_categorical=True, max_bin=max_bin,
                                    ref=d_ent)
    else:
        Path(cache).mkdir(parents=True, exist_ok=True)
        d_val = xgb.ExtMemQuantileDMatrix(IteradorParquet(validacion, y_val, Path(cache) / "validacion"),
                                          enable_categorical=True, max_bin=max_bin, ref=d_ent)
    return d_ent, d_val


class VigilarMemoria(xgb.callback.TrainingCallback):
    """Para el entrenamiento si el equipo se queda sin RAM."""

    def __init__(self, minimo_libre_gb: float, cada: int = 25):
        self.minimo_libre_gb, self.cada = minimo_libre_gb, cada
        super().__init__()

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        if epoch % self.cada == 0:
            comprobar_memoria(self.minimo_libre_gb)
        return False


def entrenar_xgb(d_ent: xgb.QuantileDMatrix, d_val: xgb.QuantileDMatrix, parametros: dict, max_arboles: int,
                 paciencia: int, callbacks: list | None = None, registro: int = 100) -> xgb.Booster:
    """Parada temprana por la logloss de validación. Devuelve el modelo recortado en su mejor árbol."""
    modelo = xgb.train(parametros, d_ent, num_boost_round=max_arboles, evals=[(d_val, "validacion")],
                       early_stopping_rounds=paciencia, verbose_eval=registro, callbacks=callbacks)
    recortado = modelo[: modelo.best_iteration + 1]
    recortado.set_attr(rondas_entrenadas=str(modelo.num_boosted_rounds()))
    return recortado


def predecir_xgb(modelo: xgb.Booster, secuencia: SecuenciaParquet) -> np.ndarray:
    """Probabilidades de todas las filas de ``secuencia``, bloque a bloque, con todos los árboles del modelo."""
    partes = [modelo.predict(xgb.DMatrix(X, feature_names=secuencia.columnas, feature_types=tipos(secuencia.columnas),
                                         enable_categorical=True))
              for X in secuencia.bloques()]
    return np.concatenate(partes).astype(np.float64) if partes else np.empty(0)
