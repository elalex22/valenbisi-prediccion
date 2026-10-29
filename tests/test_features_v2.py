import numpy as np
import pandas as pd
import pytest

from valenbisi.features import libres_en, preparar_referencia, retardos_largos
from valenbisi.features_v2 import FEATURES_V2, construir_features_v2
from valenbisi.open_meteo import VARIABLES
from valenbisi.vecinas import preparar_rejilla, vecinas_mas_cercanas

INICIO = pd.Timestamp("2024-01-08 00:00", tz="UTC")
ESTACIONES = pd.DataFrame({"estacion": [1, 2, 3], "lat": [39.47, 39.47, 39.48], "lon": [-0.38, -0.37, -0.38]})
TIEMPO = pd.DataFrame({"fecha": pd.to_datetime(["2024-01-01"]),
                       **{v: [1.0] for v in ["tmed", "tmin", "tmax", "prec", "sol"]}})


def serie(dias: int, estaciones=(1, 2, 3), semilla: int = 0) -> pd.DataFrame:
    """Fotos cada 15 min durante ``dias`` días; ``libres`` aleatorio entre 0 y 15 de 15 anclajes."""
    rng = np.random.default_rng(semilla)
    slots = pd.date_range(INICIO, periods=dias * 96, freq="15min")
    df = pd.DataFrame([(s, e) for s in slots for e in estaciones], columns=["slot_utc", "estacion"])
    libres = rng.integers(0, 16, len(df))
    return df.assign(estacion=df["estacion"].astype("int16"), libres=libres.astype("int16"),
                     bicis=(15 - libres).astype("int16"), total=np.int16(15), llena=libres == 0)


def horario(dias: int) -> pd.DataFrame:
    horas = pd.date_range(INICIO - pd.Timedelta(hours=3), periods=dias * 24 + 3, freq="h")
    return pd.DataFrame({"hora_utc": horas, **{n: np.arange(len(horas), dtype="float32")
                                               for n in VARIABLES.values()}})


def valor(df: pd.DataFrame, momento: pd.Timestamp, estacion: int) -> float:
    return float(df.loc[(df["slot_utc"] == momento) & (df["estacion"] == estacion), "libres"].iloc[0])


def test_libres_en_busca_el_desplazamiento_exacto_o_nan():
    df = serie(2)
    t = INICIO + pd.Timedelta(days=1, hours=5)
    fila = df[(df["slot_utc"] == t) & (df["estacion"] == 2)]
    referencia = preparar_referencia(df)
    assert libres_en(fila, referencia, 96)[0] == valor(df, t - pd.Timedelta(days=1), 2)
    assert np.isnan(libres_en(fila, referencia, 96 * 3)[0])     # antes del primer dato


def test_retardos_alineados_con_la_hora_del_objetivo():
    df = serie(8)
    t = INICIO + pd.Timedelta(days=7, hours=10)
    fila = df[(df["slot_utc"] == t) & (df["estacion"] == 1)]
    r = retardos_largos(fila, preparar_referencia(df)).iloc[0]

    objetivo = t + pd.Timedelta(minutes=60)
    assert r["libres_obj_24h"] == valor(df, objetivo - pd.Timedelta(days=1), 1)    # t − 23 h
    assert r["cambio_24h"] == valor(df, objetivo - pd.Timedelta(days=1), 1) - valor(df, t - pd.Timedelta(days=1), 1)
    assert r["libres_obj_7d"] == valor(df, objetivo - pd.Timedelta(days=7), 1)     # t − 7 d + 1 h
    assert r["cambio_7d"] == valor(df, objetivo - pd.Timedelta(days=7), 1) - valor(df, t - pd.Timedelta(days=7), 1)


def test_retardos_sin_la_foto_quedan_nan():
    df = serie(2)
    t = INICIO + pd.Timedelta(days=1, hours=10)
    hueco = t + pd.Timedelta(minutes=60) - pd.Timedelta(days=1)
    con_hueco = df[~((df["slot_utc"] == hueco) & (df["estacion"] == 1))]
    fila = df[(df["slot_utc"] == t) & (df["estacion"] == 1)]
    r = retardos_largos(fila, preparar_referencia(con_hueco)).iloc[0]
    assert np.isnan(r["libres_obj_24h"]) and np.isnan(r["cambio_24h"])
    assert np.isnan(r["libres_obj_7d"])                      # no hay 7 días de historia


def construir(df: pd.DataFrame, dias: int) -> pd.DataFrame:
    return construir_features_v2(df, preparar_referencia(df), preparar_rejilla(df),
                                 vecinas_mas_cercanas(ESTACIONES, k=2), TIEMPO, horario(dias), ESTACIONES)


def test_construir_features_v2_da_las_35_columnas():
    df = serie(2).assign(llena_60=False)
    out = construir(df, 2)
    assert len(FEATURES_V2) == 35 and len(set(FEATURES_V2)) == 35
    assert list(out.columns) == ["slot_utc", *FEATURES_V2, "llena_60"]
    assert len(out) == len(df)


def test_sin_fuga_cambiar_el_futuro_de_cualquier_estacion_no_cambia_las_features_de_t():
    """Ninguna de las 35 features de una fila puede depender de fotos (de ninguna estación) posteriores a t."""
    dias = 8
    original = serie(dias)
    corte = INICIO + pd.Timedelta(days=7, hours=12)
    futuro = original["slot_utc"] > corte
    alterado = original.copy()
    alterado.loc[futuro, "libres"] = (15 - alterado.loc[futuro, "libres"]).astype("int16")
    alterado.loc[futuro, "bicis"] = (15 - alterado.loc[futuro, "libres"]).astype("int16")
    alterado["llena"] = alterado["libres"].eq(0)
    # Se borra además parte del futuro, de otra estación: que falte tampoco puede influir.
    alterado = alterado.drop(alterado.index[futuro & (alterado["estacion"] == 2)])
    # Y la meteorología posterior a t también cambia.
    tiempo_alterado = horario(dias)
    tiempo_alterado.loc[tiempo_alterado["hora_utc"] > corte, list(VARIABLES.values())] = -99.0

    vecinas = vecinas_mas_cercanas(ESTACIONES, k=2)
    antes = construir_features_v2(original, preparar_referencia(original), preparar_rejilla(original), vecinas,
                                  TIEMPO, horario(dias), ESTACIONES)
    despues = construir_features_v2(alterado, preparar_referencia(alterado), preparar_rejilla(alterado), vecinas,
                                    TIEMPO, tiempo_alterado, ESTACIONES)
    pasado = lambda d: d[d["slot_utc"] <= corte].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(pasado(antes), pasado(despues))
    # El test no es trivial: justo después del corte, las vecinas y la meteorología sí cambian.
    siguiente = lambda d: d[d["slot_utc"] == corte + pd.Timedelta(hours=1)].reset_index(drop=True)  # noqa: E731
    for columna in ("vec_ocupacion", "temp"):
        assert not siguiente(antes)[columna].equals(siguiente(despues)[columna]), columna
