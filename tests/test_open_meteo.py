import json
from datetime import date
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from valenbisi.open_meteo import FEATURES_METEO, VARIABLES, leer_respuesta, tiempo_hora_pasada, url_archivo


def horario(horas: int = 6, inicio: str = "2024-03-31 00:00") -> pd.DataFrame:
    """Horas consecutivas en UTC; cada variable vale el número de la hora (0, 1, 2…) para poder seguirla."""
    hora = pd.date_range(inicio, periods=horas, freq="h", tz="UTC")
    valores = np.arange(horas, dtype="float32")
    return pd.DataFrame({"hora_utc": hora, **{nombre: valores for nombre in VARIABLES.values()}})


def slots(*momentos: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(momentos)).tz_localize("UTC"))


def test_url_pide_utc_unix_y_las_variables():
    q = parse_qs(urlparse(url_archivo(date(2022, 12, 1), date(2025, 9, 24))).query)
    assert q["timezone"] == ["GMT"] and q["timeformat"] == ["unixtime"]
    assert q["start_date"] == ["2022-12-01"] and q["end_date"] == ["2025-09-24"]
    assert q["hourly"][0].split(",") == list(VARIABLES)


def guardar(tmp_path, horas: list[int], **cambios) -> object:
    datos = {"hourly": {"time": horas, **{v: [1.0] * len(horas) for v in VARIABLES}}}
    datos["hourly"].update(cambios)
    ruta = tmp_path / "r.json"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


def test_leer_respuesta_pasa_null_a_nan_y_comprueba_las_horas(tmp_path):
    t0 = 1_700_000_000 - 1_700_000_000 % 3600
    df = leer_respuesta(guardar(tmp_path, [t0, t0 + 3600, t0 + 7200], temperature_2m=[10.5, None, 12.0]))
    assert df["hora_utc"].dt.tz is not None and len(df) == 3
    assert df["temp"].dtype == np.float32 and np.isnan(df["temp"].iloc[1])

    with pytest.raises(ValueError, match="consecutivas"):
        leer_respuesta(guardar(tmp_path, [t0, t0 + 7200]))
    with pytest.raises(ValueError, match="Faltan"):
        ruta = guardar(tmp_path, [t0])
        datos = json.loads(ruta.read_text())
        del datos["hourly"]["precipitation"]
        ruta.write_text(json.dumps(datos))
        leer_respuesta(ruta)


def test_usa_la_hora_ya_empezada_y_la_lluvia_de_las_tres_anteriores():
    # Horas 00:00 (0) a 05:00 (5). La lluvia de la hora h es la caída entre h−1 y h.
    df = tiempo_hora_pasada(slots("2024-03-31 02:00", "2024-03-31 02:45", "2024-03-31 00:30"), horario())
    assert list(df.columns) == FEATURES_METEO
    assert df["temp"].tolist() == [2.0, 2.0, 0.0]          # 02:45 usa las 02:00, nunca las 03:00
    assert df["prec_1h"].tolist() == [2.0, 2.0, 0.0]
    assert df["prec_3h"].iloc[0] == 0 + 1 + 2             # horas 00, 01 y 02
    assert np.isnan(df["prec_3h"].iloc[2])                 # a las 00:00 no hay tres horas previas


def test_huecos_quedan_nan_sin_rellenar():
    h = horario().drop(index=3)                            # falta la hora 03:00
    df = tiempo_hora_pasada(slots("2024-03-31 03:15", "2024-03-31 04:00", "2024-03-31 05:59",
                                  "2024-03-30 23:00"), h)
    assert df.iloc[0].isna().all()                         # sin dato de su hora: todo NaN
    assert df["temp"].iloc[1] == 4.0
    assert np.isnan(df["prec_3h"].iloc[1]) and np.isnan(df["prec_3h"].iloc[2])  # la suma incluiría las 03:00
    assert df.iloc[3].isna().all()                         # antes del primer dato


def test_no_usa_horas_posteriores_a_t():
    base = horario(24)
    t = slots("2024-03-31 10:00", "2024-03-31 10:59", "2024-03-31 07:30")
    antes = tiempo_hora_pasada(t, base)
    futuro = base.copy()
    columnas = list(VARIABLES.values())
    futuro.loc[futuro["hora_utc"] > pd.Timestamp("2024-03-31 10:00", tz="UTC"), columnas] = -999.0
    pd.testing.assert_frame_equal(antes, tiempo_hora_pasada(t, futuro))
    # Y sí cambia si se toca la hora usada: el test no es trivial.
    tocado = base.copy()
    tocado.loc[tocado["hora_utc"] == pd.Timestamp("2024-03-31 10:00", tz="UTC"), "temp"] = -999.0
    assert tiempo_hora_pasada(t, tocado)["temp"].iloc[0] == -999.0


def test_conserva_el_indice_de_las_filas():
    t = slots("2024-03-31 01:00", "2024-03-31 02:00")
    t.index = [10, 20]
    assert tiempo_hora_pasada(t, horario()).index.tolist() == [10, 20]


def test_llueve_1h_es_binaria_con_umbral_y_conserva_los_huecos():
    h = horario(4)
    h["prec_1h"] = np.array([0.0, 0.05, 0.1, np.nan], dtype="float32")
    df = tiempo_hora_pasada(slots("2024-03-31 00:10", "2024-03-31 01:10", "2024-03-31 02:10", "2024-03-31 03:10"), h)
    assert df["llueve_1h"].iloc[:3].tolist() == [0.0, 0.0, 1.0]   # 0,05 mm no llega al umbral; 0,1 sí
    assert np.isnan(df["llueve_1h"].iloc[3])
