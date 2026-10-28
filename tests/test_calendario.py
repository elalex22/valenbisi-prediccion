import pandas as pd
import pytest

from valenbisi.calendario import FESTIVOS, features_calendario


def slots(*locales: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(locales)).tz_localize("Europe/Madrid").tz_convert("UTC"))


def test_hora_local_y_dia_de_la_semana():
    # 2024-03-19 (martes) a las 08:45 locales son las 07:45 UTC.
    df = features_calendario(slots("2024-03-19 08:45"))
    assert df.iloc[0].to_dict() == {"minuto_del_dia": 525, "dia_semana": 1, "mes": 3, "festivo": True}


def test_el_festivo_es_el_dia_local_no_el_utc():
    # 00:30 locales del 22-01-2024 son las 23:30 UTC del 21: es San Vicente Mártir.
    # 23:30 locales del 21-01 (domingo, no festivo) son las 22:30 UTC del mismo día.
    df = features_calendario(slots("2024-01-22 00:30", "2024-01-21 23:30"))
    assert df["festivo"].tolist() == [True, False]


def test_cambio_de_hora_de_marzo():
    # El 31-03-2024 se pasa de las 02:00 a las 03:00: dos slots UTC seguidos dan 01:45 y 03:00.
    utc = pd.Series(pd.to_datetime(["2024-03-31 00:45", "2024-03-31 01:00"]).tz_localize("UTC"))
    assert features_calendario(utc)["minuto_del_dia"].tolist() == [105, 180]


def test_tabla_sin_repetidos_y_sin_el_26_de_diciembre_de_2022():
    assert FESTIVOS.is_unique
    assert pd.Timestamp("2022-12-26") not in FESTIVOS


def test_fuera_de_la_tabla_da_error():
    with pytest.raises(ValueError, match="festivos"):
        features_calendario(slots("2026-01-05 10:00"))
