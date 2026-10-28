import numpy as np
import pandas as pd
import pytest

from valenbisi.features import (
    FEATURES,
    construir_features,
    deltas_libres,
    estado_actual,
    preparar_referencia,
    tiempo_de_ayer,
)

DIA = "2024-01-15"  # lunes


def fotos(*registros) -> pd.DataFrame:
    """registros: (hora UTC 'HH:MM' del 15-01-2024, estacion, libres); 15 anclajes, todos funcionan."""
    df = pd.DataFrame(registros, columns=["hora", "estacion", "libres"])
    slot = pd.to_datetime(DIA + " " + df.pop("hora")).dt.tz_localize("UTC").astype("datetime64[us, UTC]")
    return pd.DataFrame({
        "slot_utc": slot,
        "estacion": df["estacion"].astype("int16"),
        "bicis": (15 - df["libres"]).astype("int16"),
        "libres": df["libres"].astype("int16"),
        "total": pd.Series(15, index=df.index, dtype="int16"),
        "llena": df["libres"].eq(0),
    })


def deltas(filas: pd.DataFrame, referencia: pd.DataFrame | None = None) -> dict:
    """{(hora, estacion): (delta_15, delta_30, delta_60)} con NaN como None."""
    d = deltas_libres(filas, preparar_referencia(filas if referencia is None else referencia))
    return {
        (s.strftime("%H:%M"), int(e)): tuple(None if np.isnan(v) else float(v) for v in fila)
        for s, e, fila in zip(filas["slot_utc"], filas["estacion"], d.to_numpy())
    }


TIEMPO = pd.DataFrame({
    "fecha": pd.to_datetime(["2024-01-14", "2024-01-15"]),
    "tmed": [10.0, 20.0], "tmin": [5.0, 15.0], "tmax": [15.0, 25.0], "prec": [0.0, 30.0], "sol": [8.0, 1.0],
}).astype({v: "float32" for v in ["tmed", "tmin", "tmax", "prec", "sol"]})
ESTACIONES = pd.DataFrame({"estacion": [1, 2], "lat": [39.47, 39.48], "lon": [-0.37, -0.38]})


def test_deltas_con_todas_las_fotos():
    df = fotos(("10:00", 1, 10), ("10:15", 1, 8), ("10:30", 1, 5), ("10:45", 1, 4), ("11:00", 1, 1))
    assert deltas(df)[("11:00", 1)] == (-3.0, -4.0, -9.0)
    # A las 10:15 hay t−15 pero no t−30 ni t−60.
    assert deltas(df)[("10:15", 1)] == (-2.0, None, None)


def test_si_falta_la_foto_queda_nan_y_no_se_rellena_con_otra():
    # Falta la de las 10:45: la de las 11:00 no tiene t−15, aunque haya t−30.
    df = fotos(("10:00", 1, 10), ("10:15", 1, 8), ("10:30", 1, 5), ("11:00", 1, 1))
    assert deltas(df)[("11:00", 1)] == (None, -4.0, -9.0)


def test_cada_estacion_busca_solo_sus_fotos():
    df = fotos(("10:00", 1, 10), ("10:15", 2, 3))
    assert deltas(df)[("10:15", 2)] == (None, None, None)


def test_las_fotos_sin_etiqueta_tambien_sirven_de_pasado():
    referencia = fotos(("10:00", 1, 10), ("10:15", 1, 8))
    filas = referencia.iloc[[1]]  # la de las 10:00 no tenía t+60, pero su libres es válido
    assert deltas(filas, referencia)[("10:15", 1)] == (-2.0, None, None)


def test_referencia_con_fotos_repetidas_da_error():
    with pytest.raises(ValueError, match="mismo"):
        preparar_referencia(fotos(("10:00", 1, 1), ("10:00", 1, 2)))


def test_ocupacion_con_anclajes_averiados():
    # 10 bicis, 0 libres, 15 de total: 5 anclajes averiados. Está llena y la ocupación es 1.
    df = fotos(("10:00", 1, 0)).assign(bicis=np.int16(10))
    assert estado_actual(df)["ocupacion"].iloc[0] == 1.0


def test_el_tiempo_es_el_del_dia_local_anterior():
    # 23:30 UTC del 14-01 son las 00:30 del 15 en València: el «ayer» es el 14, no el 13.
    slot = pd.Series(pd.to_datetime(["2024-01-14 23:30", "2024-01-15 22:00"]).tz_localize("UTC"))
    df = tiempo_de_ayer(slot, TIEMPO)
    assert df["tmed_ayer"].tolist() == [10.0, 10.0]
    # El 16-01 local (23:00 del 15 UTC) usa el 15; y un día sin datos queda NaN.
    slot = pd.Series(pd.to_datetime(["2024-01-15 23:00", "2024-01-14 01:00"]).tz_localize("UTC"))
    assert tiempo_de_ayer(slot, TIEMPO)["prec_ayer"].tolist()[0] == 30.0
    assert np.isnan(tiempo_de_ayer(slot, TIEMPO)["prec_ayer"].iloc[1])


def test_construir_features_da_todas_las_columnas():
    df = fotos(("10:00", 1, 10), ("10:15", 1, 8), ("10:15", 2, 0)).assign(llena_60=[False, True, True])
    out = construir_features(df, preparar_referencia(df), TIEMPO, ESTACIONES)

    assert list(out.columns) == ["slot_utc", *FEATURES, "llena_60"]
    assert out["lat"].tolist() == pytest.approx([39.47, 39.47, 39.48])
    assert out["llena_60"].tolist() == [False, True, True]


def test_sin_fuga_cambiar_el_futuro_no_cambia_las_features_de_t():
    """Las features de una fila solo pueden depender de fotos con slot_utc ≤ t."""
    rng = np.random.default_rng(0)
    horas = [f"{h:02d}:{m:02d}" for h in range(9, 13) for m in (0, 15, 30, 45)]
    registros = [(h, e, int(rng.integers(0, 16))) for h in horas for e in (1, 2)]
    original = fotos(*registros)
    corte = pd.Timestamp(f"{DIA} 11:00", tz="UTC")

    futuro = original["slot_utc"] > corte
    alterado = original.copy()
    alterado.loc[futuro, "libres"] = 15 - alterado.loc[futuro, "libres"]
    alterado.loc[futuro, "bicis"] = 15 - alterado.loc[futuro, "libres"]
    alterado["llena"] = alterado["libres"].eq(0)
    # Y además se borra parte del futuro: tampoco puede influir que falte.
    alterado = alterado.drop(alterado.index[futuro & (alterado["estacion"] == 2)])

    antes = construir_features(original, preparar_referencia(original), TIEMPO, ESTACIONES)
    despues = construir_features(alterado, preparar_referencia(alterado), TIEMPO, ESTACIONES)
    pasado_antes = antes[antes["slot_utc"] <= corte].reset_index(drop=True)
    pasado_despues = despues[despues["slot_utc"] <= corte].reset_index(drop=True)
    pd.testing.assert_frame_equal(pasado_antes, pasado_despues)
    # Comprobación de que el test sirve: las filas del futuro sí cambian.
    assert not antes[antes["slot_utc"] > corte]["libres"].reset_index(drop=True).equals(
        despues[despues["slot_utc"] > corte]["libres"].reset_index(drop=True))
