import numpy as np
import pandas as pd
import pytest

from valenbisi.vecinas import (
    FALTA,
    FEATURES_VECINAS,
    distancias_km,
    features_vecinas,
    preparar_rejilla,
    vecinas_mas_cercanas,
)

# Cuatro estaciones en fila hacia el este: la 1 está a 0, la 2 a ~0,9 km, la 3 a ~1,7 km y la 4 a ~4,3 km.
ESTACIONES = pd.DataFrame({"estacion": [1, 2, 3, 4], "lat": [39.47] * 4,
                           "lon": [-0.40, -0.39, -0.38, -0.35]})


def test_distancia_haversine():
    d = distancias_km([0.0, 39.47], [0.0, -0.37], [1.0, 39.47], [0.0, -0.37])
    assert d[0, 0] == pytest.approx(111.19, abs=0.01)   # 1 grado de latitud
    assert d[1, 1] == pytest.approx(0.0)


def test_vecinas_por_distancia_sin_contarse_a_si_misma():
    tabla = vecinas_mas_cercanas(ESTACIONES, k=2)
    assert tabla.shape[1] == 2
    assert tabla[1].tolist() == [2, 3]
    assert tabla[2].tolist() == [1, 3]   # la 1 y la 3 están a la misma distancia: gana la primera
    assert tabla[4].tolist() == [3, 2]
    assert (tabla[5] == FALTA).all()     # estación que no existe


def test_con_k_mayor_que_las_demas_sobran_huecos():
    tabla = vecinas_mas_cercanas(ESTACIONES.iloc[:2], k=3)
    assert tabla[1].tolist() == [2, FALTA, FALTA]


def test_estacion_nueva_busca_entre_las_conocidas_sin_ser_vecina_de_ellas():
    nueva = pd.DataFrame({"estacion": [9], "lat": [39.47], "lon": [-0.355]})
    tabla = vecinas_mas_cercanas(nueva, k=2, candidatas=ESTACIONES)
    assert tabla[9].tolist() == [4, 3]
    assert (tabla[4] == FALTA).all()     # las conocidas no se recalculan aquí


def fotos(*registros) -> pd.DataFrame:
    """registros: (hora UTC 'HH:MM' del 15-01-2024, estacion, libres, bicis)."""
    df = pd.DataFrame(registros, columns=["hora", "estacion", "libres", "bicis"])
    slot = pd.to_datetime("2024-01-15 " + df.pop("hora")).dt.tz_localize("UTC")
    return df.assign(slot_utc=slot).astype({"estacion": "int16", "libres": "int16", "bicis": "int16"})


def filas(*registros) -> pd.DataFrame:
    """registros: (hora 'HH:MM', estacion)."""
    df = pd.DataFrame(registros, columns=["hora", "estacion"])
    return pd.DataFrame({"slot_utc": pd.to_datetime("2024-01-15 " + df["hora"]).dt.tz_localize("UTC"),
                         "estacion": df["estacion"].astype("int16")})


def test_rejilla_con_fotos_repetidas_da_error():
    with pytest.raises(ValueError, match="mismo"):
        preparar_rejilla(fotos(("10:00", 1, 3, 7), ("10:00", 1, 4, 6)))


def test_features_de_las_vecinas_en_t_y_t_menos_60():
    rejilla = preparar_rejilla(fotos(
        ("09:00", 2, 6, 4), ("09:00", 3, 2, 8),
        ("10:00", 1, 9, 1), ("10:00", 2, 0, 10), ("10:00", 3, 5, 15),
    ))
    vecinas = vecinas_mas_cercanas(ESTACIONES.iloc[:3], k=2)   # la 1 tiene de vecinas a la 2 y la 3
    df = features_vecinas(filas(("10:00", 1)), rejilla, vecinas)

    assert list(df.columns) == FEATURES_VECINAS
    fila = df.iloc[0]
    assert fila["vec_ocupacion"] == pytest.approx((10 / 10 + 15 / 20) / 2)
    assert fila["vec_frac_llenas"] == 0.5
    assert fila["vec_libres_min"] == 0
    assert fila["vec_delta_libres_60"] == pytest.approx(((0 - 6) + (5 - 2)) / 2)


def test_vecinas_sin_foto_se_ignoran_y_sin_ninguna_queda_nan():
    rejilla = preparar_rejilla(fotos(("10:00", 2, 4, 6), ("10:15", 1, 1, 1), ("11:00", 3, 3, 3)))
    vecinas = vecinas_mas_cercanas(ESTACIONES.iloc[:3], k=2)
    df = features_vecinas(filas(("10:00", 1), ("10:15", 1), ("09:00", 1), ("12:00", 1)), rejilla, vecinas)

    assert df["vec_ocupacion"].iloc[0] == pytest.approx(0.6)       # solo la 2 tiene foto a las 10:00
    assert df["vec_libres_min"].iloc[0] == 4
    assert np.isnan(df["vec_delta_libres_60"].iloc[0])             # la 2 no tiene foto a las 09:00
    assert df.iloc[1].isna().all()                                 # a las 10:15 ninguna vecina tiene foto
    assert df.iloc[2].isna().all() and df.iloc[3].isna().all()    # antes y después de la rejilla


def test_estacion_sin_vecinas_da_nan():
    rejilla = preparar_rejilla(fotos(("10:00", 1, 1, 1)))
    vecinas = np.full((1000, 2), FALTA, dtype="int32")
    assert features_vecinas(filas(("10:00", 1)), rejilla, vecinas).iloc[0].isna().all()
