"""Regresión logística por tramos: indicadores, estación nueva, guardado y ajuste por bloques (datos inventados)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from valenbisi.logistica import CORTES_LIBRES, Logistica, Tramos, ajustar, ajustar_tramos, cortes_cuantiles

COLUMNAS = ["llena", "libres", "ocupacion", "estacion", "minuto_del_dia", "dia_semana", "mes", "festivo",
            "cambio_7d"]


def datos(n: int, semilla: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Filas inventadas con las columnas de COLUMNAS y una etiqueta que depende de libres, hora y estación."""
    rng = np.random.default_rng(semilla)
    libres = rng.integers(0, 20, n)
    estacion = rng.choice([3, 7, 11, 20], n)
    minuto = rng.integers(0, 96, n) * 15
    dia = rng.integers(0, 7, n)
    cambio = np.where(rng.random(n) < 0.1, np.nan, rng.normal(0, 3, n))
    X = np.column_stack([libres == 0, libres, rng.random(n), estacion, minuto, dia, rng.integers(1, 13, n),
                         rng.random(n) < 0.05, cambio]).astype(np.float32)
    z = -3 + 2.5 * (libres == 0) - 0.15 * libres + 0.8 * (estacion == 7) + 0.5 * (minuto > 600) \
        + 0.4 * np.nan_to_num(cambio < -2) + 0.6 * np.isnan(cambio)
    y = rng.random(n) < 1 / (1 + np.exp(-z))
    return X, y


def tramos_de(X: np.ndarray) -> Tramos:
    return ajustar_tramos(COLUMNAS, lambda c: X[:, COLUMNAS.index(c)])


def test_cortes_por_deciles_unicos_y_sin_nan():
    assert cortes_cuantiles(np.arange(1, 101, dtype=np.float32)) == pytest.approx(
        [10.9, 20.8, 30.7, 40.6, 50.5, 60.4, 70.3, 80.2, 90.1])
    assert cortes_cuantiles(np.array([5, 5, 5, np.nan], dtype=np.float32)) == [5.0]
    assert cortes_cuantiles(np.array([np.nan], dtype=np.float32)) == []


def test_libres_va_por_valor_exacto_hasta_10_y_luego_11_15_y_16_o_mas():
    X, _ = datos(200)
    t = tramos_de(X)
    assert t.cortes["libres"] == [float(c) for c in CORTES_LIBRES]
    filas = np.repeat(X[:1], 9, axis=0)
    filas[:, 1] = [0, 1, 5, 10, 11, 15, 16, 40, np.nan]
    nombres = t.nombres()
    activa = [nombres[i] for i in t.indices(filas)[:, 1]]
    assert activa == ["libres tramo 0", "libres tramo 1", "libres tramo 5", "libres tramo 10", "libres tramo 11",
                      "libres tramo 11", "libres tramo 12", "libres tramo 12", "libres falta"]


def test_nan_va_a_su_tramo_falta_y_solo_a_ese():
    X, _ = datos(500)
    t = tramos_de(X)
    j = COLUMNAS.index("cambio_7d")
    M = t.matriz(X).toarray()
    nombres = t.nombres()
    del_cambio = [i for i, n in enumerate(nombres) if n.startswith("cambio_7d")]
    falta = nombres.index("cambio_7d falta")
    np.testing.assert_array_equal(M[:, falta] == 1, np.isnan(X[:, j]))
    np.testing.assert_array_equal(M[:, del_cambio].sum(axis=1), 1)      # siempre un tramo, y solo uno


def test_cada_fila_activa_un_indicador_por_variable_y_una_estacion_nueva_da_ceros():
    X, _ = datos(300)
    t = tramos_de(X)
    assert t.estaciones == [3, 7, 11, 20]
    assert len(t.nombres()) == t.ancho
    M = t.matriz(X)
    variables = [v for v, _ in t.bloques() if not (v in ("llena", "festivo"))]
    activos_binarias = X[:, COLUMNAS.index("llena")] + X[:, COLUMNAS.index("festivo")]
    np.testing.assert_array_equal(M.sum(axis=1).A1, len(variables) + activos_binarias)
    nueva = X[:1].copy()
    nueva[0, COLUMNAS.index("estacion")] = 999
    estaciones = [i for i, n in enumerate(t.nombres()) if n.startswith("estacion ")]
    assert t.matriz(nueva).toarray()[0, estaciones].sum() == 0
    assert t.matriz(X[:1]).toarray()[0, estaciones].sum() == 1


def test_hora_por_tipo_de_dia():
    X, _ = datos(50)
    t = tramos_de(X)
    filas = np.repeat(X[:1], 3, axis=0)
    filas[:, COLUMNAS.index("minuto_del_dia")] = [0, 8 * 60 + 45, 23 * 60 + 45]
    filas[:, COLUMNAS.index("dia_semana")] = [0, 5, 6]
    nombres = t.nombres()
    hora = [nombres[i] for i in t.indices(filas)[:, -3]]
    assert hora == ["hora 0 laborable", "hora 8 fin de semana", "hora 23 fin de semana"]


def test_el_modelo_guardado_predice_exactamente_lo_mismo(tmp_path):
    X, _ = datos(300)
    t = tramos_de(X)
    modelo = Logistica(t, np.random.default_rng(1).normal(0, 1, t.ancho), -1.234567890123)
    modelo.guardar(tmp_path / "m.json")
    np.testing.assert_array_equal(Logistica.cargar(tmp_path / "m.json").predecir(X), modelo.predecir(X))


def ajuste_por_bloques(X, y, alpha, semilla=42, n_bloques=4, max_epocas=8, X_val=None, y_val=None, **kwargs):
    t = tramos_de(X)
    cortes = np.linspace(0, len(y), n_bloques + 1).astype(int)
    X_val, y_val = (X, y) if X_val is None else (X_val, y_val)
    return t, ajustar(t, lambda g: (X[cortes[g]:cortes[g + 1]], y[cortes[g]:cortes[g + 1]]), n_bloques,
                      lambda: [X_val[:1000], X_val[1000:]], y_val, alpha=alpha, semilla=semilla,
                      max_epocas=max_epocas, **kwargs)


def test_el_sgd_por_bloques_se_acerca_a_la_logistica_con_la_matriz_entera():
    X, y = datos(50_000)
    alpha = 1e-6       # la R2: con el paso por defecto de scikit-learn, el SGD se quedaba lejos
    t, ajuste = ajuste_por_bloques(X, y, alpha, paciencia=8)
    M = t.matriz(X)
    # Misma función objetivo: SGD minimiza logloss media + alpha/2·|w|²; LogisticRegression, C·Σlogloss + |w|²/2.
    exacta = LogisticRegression(C=1 / (alpha * len(y)), max_iter=2000).fit(M, y)
    p_exacta = exacta.predict_proba(M)[:, 1]
    p_sgd = ajuste.modelo.predecir(X)
    assert abs(log_loss(y, p_sgd) - log_loss(y, p_exacta)) < 0.001
    assert np.abs(p_sgd - p_exacta).mean() < 0.005


def test_parada_temprana_se_queda_con_la_mejor_epoca():
    X, y = datos(4000)
    X_val, y_val = datos(3000, semilla=1)
    vistas = []
    _, ajuste = ajuste_por_bloques(X, y, 1e-6, max_epocas=6, X_val=X_val, y_val=y_val,
                                   al_acabar_epoca=lambda e, ll: vistas.append((e, ll)))
    assert [e for e, _ in vistas] == list(range(1, len(ajuste.logloss_por_epoca) + 1))
    assert len(ajuste.logloss_por_epoca) <= 6
    # Con paciencia 1, para en la primera época que no mejora (o al llegar al tope).
    h = ajuste.logloss_por_epoca
    assert all(h[i] < min(h[:i]) for i in range(1, len(h) - 1))
    assert log_loss(y_val, ajuste.modelo.predecir(X_val)) == pytest.approx(min(h))
    assert ajuste.mejor_epoca == int(np.argmin(h)) + 1


def test_la_semilla_solo_cambia_el_barajado():
    X, y = datos(3000)
    _, a = ajuste_por_bloques(X, y, 1e-5, semilla=42, max_epocas=2)
    _, b = ajuste_por_bloques(X, y, 1e-5, semilla=42, max_epocas=2)
    _, c = ajuste_por_bloques(X, y, 1e-5, semilla=43, max_epocas=2)
    np.testing.assert_array_equal(a.modelo.coef, b.modelo.coef)
    assert not np.array_equal(a.modelo.coef, c.modelo.coef)
