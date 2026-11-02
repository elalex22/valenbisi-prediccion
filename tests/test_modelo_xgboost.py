"""XGBoost por bloques: mismo modelo que con la matriz entera, estación categórica y recorte (datos inventados)."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost as xgb

from valenbisi.modelo import SecuenciaParquet
from valenbisi.modelo_xgboost import PARAMETROS_XGB, entrenar_xgb, matrices, predecir_xgb, tipos

COLUMNAS = ["libres", "ocupacion", "estacion", "cambio_7d"]
PARAMS = {**PARAMETROS_XGB, "min_child_weight": 1, "nthread": 2, "eta": 0.2, "grow_policy": "lossguide",
          "max_leaves": 15, "max_depth": 0, "seed": 42}


def parquet(ruta, n: int = 4000, semilla: int = 0) -> np.ndarray:
    """Parquet con COLUMNAS, una columna «conjunto» (0 entrenamiento, 1 validación) y llena_60; 4 row groups."""
    rng = np.random.default_rng(semilla)
    estacion = rng.choice([3, 7, 11, 20], n)
    cambio = np.where(rng.random(n) < 0.1, np.nan, rng.normal(0, 3, n))
    libres = rng.integers(0, 20, n)
    z = -2 - 0.2 * libres + 1.5 * (estacion == 7) - 1.0 * (estacion == 11) + 0.8 * np.isnan(cambio)
    y = rng.random(n) < 1 / (1 + np.exp(-z))
    tabla = pa.table({"libres": libres.astype("int16"), "ocupacion": rng.random(n).astype("float32"),
                      "estacion": estacion.astype("int16"), "cambio_7d": cambio.astype("float32"),
                      "conjunto": (np.arange(n) % 4 == 0).astype("int8"), "llena_60": y})
    pq.write_table(tabla, ruta, row_group_size=n // 4)
    return y


def secuencias(ruta):
    return (SecuenciaParquet(ruta, COLUMNAS, [("conjunto", "==", 0)]),
            SecuenciaParquet(ruta, COLUMNAS, [("conjunto", "==", 1)]))


def etiquetas(s: SecuenciaParquet) -> np.ndarray:
    return s.leer(["llena_60"])["llena_60"].to_numpy()


def test_la_estacion_es_categorica():
    assert tipos(["libres", "estacion", "lat"]) == ["q", "c", "q"]


def test_por_bloques_da_el_mismo_modelo_que_la_matriz_entera(tmp_path):
    parquet(tmp_path / "f.parquet")
    ent, val = secuencias(tmp_path / "f.parquet")
    y_ent, y_val = etiquetas(ent), etiquetas(val)
    d_ent, d_val = matrices(ent, y_ent, val, y_val)
    por_bloques = entrenar_xgb(d_ent, d_val, PARAMS, max_arboles=60, paciencia=10, registro=0)

    X_ent, X_val = ent[0:len(ent)], val[0:len(val)]
    entera_ent = xgb.QuantileDMatrix(X_ent, label=y_ent, feature_names=COLUMNAS, feature_types=tipos(COLUMNAS),
                                     enable_categorical=True, max_bin=PARAMS["max_bin"])
    entera_val = xgb.QuantileDMatrix(X_val, label=y_val, feature_names=COLUMNAS, feature_types=tipos(COLUMNAS),
                                     enable_categorical=True, max_bin=PARAMS["max_bin"], ref=entera_ent)
    entera = entrenar_xgb(entera_ent, entera_val, PARAMS, max_arboles=60, paciencia=10, registro=0)
    assert por_bloques.num_boosted_rounds() == entera.num_boosted_rounds()
    np.testing.assert_array_equal(predecir_xgb(por_bloques, val), predecir_xgb(entera, val))


def test_parte_la_estacion_por_conjuntos_y_una_nueva_no_falla(tmp_path):
    parquet(tmp_path / "f.parquet")
    ent, val = secuencias(tmp_path / "f.parquet")
    modelo = entrenar_xgb(*matrices(ent, etiquetas(ent), val, etiquetas(val)), PARAMS, max_arboles=30,
                          paciencia=10, registro=0)
    assert modelo.feature_types == ["q", "q", "c", "q"]
    arboles = modelo.trees_to_dataframe()
    cortes = arboles[arboles["Feature"] == "estacion"]["Category"].dropna()
    assert len(cortes) > 0
    # Los códigos son los números de estación: las particiones reparten de 0 al máximo visto (20).
    assert set().union(*map(set, cortes)) <= set(range(21))
    X = val[0:5].copy()
    X[:, COLUMNAS.index("estacion")] = 999
    p = modelo.predict(xgb.DMatrix(X, feature_names=COLUMNAS, feature_types=tipos(COLUMNAS), enable_categorical=True))
    assert np.all((p > 0) & (p < 1))


def test_se_recorta_en_el_mejor_arbol_y_el_guardado_predice_lo_mismo(tmp_path):
    parquet(tmp_path / "f.parquet")
    ent, val = secuencias(tmp_path / "f.parquet")
    d_ent, d_val = matrices(ent, etiquetas(ent), val, etiquetas(val))
    completo = xgb.train(PARAMS, d_ent, num_boost_round=300, evals=[(d_val, "v")], early_stopping_rounds=10,
                         verbose_eval=False)
    modelo = entrenar_xgb(d_ent, d_val, PARAMS, max_arboles=300, paciencia=10, registro=0)
    assert modelo.num_boosted_rounds() == completo.best_iteration + 1
    assert int(modelo.attr("rondas_entrenadas")) == completo.num_boosted_rounds() > modelo.num_boosted_rounds()
    dm = xgb.DMatrix(val[0:len(val)], feature_names=COLUMNAS, feature_types=tipos(COLUMNAS), enable_categorical=True)
    np.testing.assert_array_equal(predecir_xgb(modelo, val),
                                  completo.predict(dm, iteration_range=(0, completo.best_iteration + 1)))
    modelo.save_model(str(tmp_path / "m.ubj"))
    np.testing.assert_array_equal(predecir_xgb(xgb.Booster(model_file=str(tmp_path / "m.ubj")), val),
                                  predecir_xgb(modelo, val))


def test_sin_pesos_de_clase():
    assert "scale_pos_weight" not in PARAMETROS_XGB and PARAMETROS_XGB["objective"] == "binary:logistic"
    assert PARAMETROS_XGB["min_child_weight"] == pytest.approx(50)


def test_la_validacion_en_disco_da_el_mismo_modelo(tmp_path):
    parquet(tmp_path / "f.parquet", n=20_000)
    ent, val = secuencias(tmp_path / "f.parquet")
    y_ent, y_val = etiquetas(ent), etiquetas(val)
    en_memoria = entrenar_xgb(*matrices(ent, y_ent, val, y_val), PARAMS, max_arboles=200, paciencia=10, registro=0)
    d_ent, d_val = matrices(ent, y_ent, val, y_val, cache=tmp_path / "cache")
    assert isinstance(d_val, xgb.ExtMemQuantileDMatrix) and any((tmp_path / "cache").iterdir())
    en_disco = entrenar_xgb(d_ent, d_val, PARAMS, max_arboles=200, paciencia=10, registro=0)
    assert en_disco.attr("rondas_entrenadas") == en_memoria.attr("rondas_entrenadas")
    assert en_disco.num_boosted_rounds() == en_memoria.num_boosted_rounds()
    np.testing.assert_array_equal(predecir_xgb(en_disco, val), predecir_xgb(en_memoria, val))
