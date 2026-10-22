import pandas as pd
import pytest

from fabrica import csv_antiguo, csv_nuevo, escribir_zip, nombre_foto
from valenbisi.lectura import COLUMNAS, leer_csv, leer_zip, ts_desde_nombre, zips_en_orden

FILAS = [
    (1, "T", 5, 10, 15, "01/03/2023 10:05:00"),
    (2, "F", 0, 0, 20, "01/03/2023 09:00:00"),
]


def test_ts_desde_nombre():
    assert ts_desde_nombre("valenbici_27-10-2024_02-15-01.csv") == pd.Timestamp("2024-10-27 02:15:01")


def test_ts_desde_nombre_rechaza_nombres_raros():
    with pytest.raises(ValueError):
        ts_desde_nombre("README.md")


@pytest.mark.parametrize("fabricar", [csv_antiguo, csv_nuevo])
def test_los_dos_esquemas_dan_las_mismas_columnas_y_valores(fabricar):
    ts = pd.Timestamp("2023-03-01 10:15:00")
    df = leer_csv(fabricar(FILAS).encode("utf-8"), ts)

    assert list(df.columns) == COLUMNAS
    assert df["estacion"].tolist() == [1, 2]
    assert df["activa"].tolist() == [True, False]
    assert df["bicis"].tolist() == [5, 0]
    assert df["libres"].tolist() == [10, 0]
    assert df["total"].tolist() == [15, 20]
    assert (df["ts"] == ts).all()
    assert df["fecha_actualizacion"].iloc[0] == pd.Timestamp("2023-03-01 10:05:00")
    assert df["lat"].iloc[0] == pytest.approx(39.47)
    assert df["lon"].iloc[0] == pytest.approx(-0.37)
    assert df["direccion"].iloc[0] == "Calle 1"


def test_los_dos_esquemas_dan_los_mismos_tipos():
    ts = pd.Timestamp("2023-03-01 10:15:00")
    antiguo = leer_csv(csv_antiguo(FILAS).encode("utf-8"), ts)
    nuevo = leer_csv(csv_nuevo(FILAS).encode("utf-8"), ts)
    pd.testing.assert_frame_equal(antiguo, nuevo)


def test_csv_solo_con_cabecera_es_foto_vacia():
    df = leer_csv(csv_nuevo([]).encode("utf-8"), pd.Timestamp("2024-01-01"))
    assert df.empty
    assert list(df.columns) == COLUMNAS


def test_leer_zip_cuenta_fotos_vacias_y_calcula_utc(carpeta_raw):
    ruta = escribir_zip(carpeta_raw / "15-01-2024.zip", [
        (nombre_foto("15-01-2024", "10-00-01"), csv_nuevo(FILAS)),
        (nombre_foto("15-01-2024", "10-15-01"), csv_nuevo([])),
    ])
    df, info = leer_zip(ruta)

    assert info["fotos"] == 2
    assert info["fotos_vacias"] == 1
    assert len(df) == 2
    # En invierno, Madrid es UTC+1.
    assert (df["ts_utc"] == pd.Timestamp("2024-01-15 09:00:01", tz="UTC")).all()


def test_cambio_de_hora_de_octubre(carpeta_raw):
    """Último domingo de octubre: la franja 02:00-02:59 ocurre dos veces.

    La segunda pasada repite incluso el nombre de fichero. Esas fotos quedan
    sin ts_utc (la limpieza las descarta); el resto se convierte bien.
    """
    fecha = "27-10-2024"
    fila = [(1, "T", 5, 10, 15, "27/10/2024 01:40:00")]
    ruta = escribir_zip(carpeta_raw / f"{fecha}.zip", [
        (nombre_foto(fecha, "01-45-00"), csv_nuevo(fila)),
        (nombre_foto(fecha, "02-00-00"), csv_nuevo(fila)),
        (nombre_foto(fecha, "02-30-00"), csv_nuevo(fila)),
        (nombre_foto(fecha, "02-00-00"), csv_nuevo(fila)),  # segunda pasada, mismo nombre
        (nombre_foto(fecha, "02-30-00"), csv_nuevo(fila)),
        (nombre_foto(fecha, "03-00-00"), csv_nuevo(fila)),
    ])
    df, info = leer_zip(ruta)

    # Se leen las 6 fotos, aunque dos nombres estén repetidos.
    assert info["fotos"] == 6
    assert len(df) == 6
    # La hora local se conserva tal cual, con las 02:00 repetidas.
    assert (df["ts"].dt.strftime("%H:%M") == "02:00").sum() == 2

    repetidas = df["ts"].dt.hour == 2
    assert df.loc[repetidas, "ts_utc"].isna().all()
    assert info["fotos_hora_repetida"] == 2  # horas locales distintas: 02:00 y 02:30

    resto = df.loc[~repetidas, "ts_utc"]
    assert resto.tolist() == [
        pd.Timestamp("2024-10-26 23:45:00", tz="UTC"),  # 01:45 CEST (UTC+2)
        pd.Timestamp("2024-10-27 02:00:00", tz="UTC"),  # 03:00 CET (UTC+1)
    ]
    assert resto.is_monotonic_increasing and resto.is_unique


def test_hora_inexistente_de_marzo_se_desplaza_y_se_cuenta(carpeta_raw):
    fecha = "30-03-2025"
    fila = [(1, "T", 5, 10, 15, "30/03/2025 01:40:00")]
    ruta = escribir_zip(carpeta_raw / f"{fecha}.zip", [
        (nombre_foto(fecha, "01-45-00"), csv_nuevo(fila)),
        (nombre_foto(fecha, "02-15-00"), csv_nuevo(fila)),  # no existe en Madrid
    ])
    df, info = leer_zip(ruta)

    assert info["fotos_hora_inexistente"] == 1
    assert df["ts_utc"].tolist() == [
        pd.Timestamp("2025-03-30 00:45:00", tz="UTC"),
        pd.Timestamp("2025-03-30 01:00:00", tz="UTC"),  # desplazada a las 03:00 CEST
    ]


def test_zips_en_orden_cronologico_no_alfabetico(carpeta_raw):
    for nombre in ["01-01-2025", "31-12-2024", "15-06-2024"]:
        escribir_zip(carpeta_raw / f"{nombre}.zip", [])
    assert [p.stem for p in zips_en_orden(carpeta_raw)] == ["15-06-2024", "31-12-2024", "01-01-2025"]
