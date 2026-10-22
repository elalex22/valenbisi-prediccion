import json

import pandas as pd

from fabrica import csv_antiguo, csv_nuevo, escribir_zip, nombre_foto
from valenbisi.dataset import COLUMNAS_SALIDA, construir_dataset, formatear_resumen


def test_extremo_a_extremo(carpeta_raw, tmp_path):
    # Zip con esquema antiguo: la estación 299, una inactiva y una normal.
    escribir_zip(carpeta_raw / "21-05-2023.zip", [
        (nombre_foto("21-05-2023", "10-00-01"), csv_antiguo([
            (299, "T", 2, 0, 2, "21/05/2023 09:55:00"),
            (1, "F", 5, 10, 15, "21/05/2023 09:55:00"),
            (2, "T", 15, 0, 15, "21/05/2023 09:55:00"),
        ])),
    ])
    # Zip con esquema nuevo y fotos de dos días (como en los datos reales):
    # una foto vacía, una congelada y una con 0 bicis y 0 huecos.
    escribir_zip(carpeta_raw / "23-05-2023.zip", [
        (nombre_foto("22-05-2023", "10-00-01"), csv_nuevo([
            (1, "T", 5, 10, 15, "22/05/2023 09:55:00"),
            (2, "T", 0, 0, 15, "22/05/2023 09:55:00"),
        ])),
        (nombre_foto("22-05-2023", "10-15-01"), csv_nuevo([
            (1, "T", 5, 10, 15, "22/05/2023 09:55:00"),
            (2, "T", 0, 0, 15, "22/05/2023 09:55:00"),
        ])),  # idéntica a la anterior
        (nombre_foto("23-05-2023", "10-00-01"), csv_nuevo([])),
        (nombre_foto("23-05-2023", "10-15-01"), csv_nuevo([
            (1, "T", 4, 11, 15, "23/05/2023 10:10:00"),
            (2, "T", 3, 12, 15, "20/05/2023 10:10:00"),  # más de 24 h
        ])),
    ])
    salida = tmp_path / "processed"
    resumen = construir_dataset(carpeta_raw, salida)

    datos = pd.read_parquet(salida / "valenbisi_limpio.parquet")
    assert list(datos.columns) == COLUMNAS_SALIDA
    assert datos["ts_utc"].is_monotonic_increasing
    assert list(zip(datos["ts"].dt.strftime("%d %H:%M"), datos["estacion"], datos["llena"])) == [
        ("21 10:00", 2, True),
        ("22 10:00", 1, False),
        ("23 10:15", 1, False),
    ]

    estaciones = pd.read_parquet(salida / "estaciones.parquet")
    assert estaciones["estacion"].tolist() == [1, 2]
    assert estaciones.loc[0, "direccion"] == "Calle 1"

    assert json.loads((salida / "resumen_limpieza.json").read_text(encoding="utf-8")) == resumen
    assert resumen["fotos"] == 5
    assert resumen["fotos_vacias"] == 1
    assert resumen["estaciones"] == 2
    assert resumen["filas"] == 3
    assert resumen["porcentaje_llena"] == round(100 / 3, 2)
    eliminadas = {p["regla"]: p["eliminadas"] for p in resumen["reglas"]}
    assert eliminadas["Estación de pruebas (299)"] == 1
    assert eliminadas["Registros inactivos"] == 1
    assert eliminadas["Actualización con más de 24 h"] == 1
    assert eliminadas["Foto idéntica a la anterior"] == 2
    assert eliminadas["0 bicis y 0 huecos libres"] == 1

    texto = formatear_resumen(resumen)
    assert "Duplicados (ts_utc, estación)" in texto
    assert "33,33 %" in texto


def test_formatear_resumen_usa_punto_para_los_miles():
    resumen = {
        "zips": 1, "fotos": 91989, "fotos_vacias": 1212, "fotos_hora_repetida": 0,
        "fotos_hora_inexistente": 0,
        "reglas": [{"regla": "Registros inactivos", "antes": 24959284, "eliminadas": 47311, "despues": 24911973}],
        "periodo_local": ["2022-12-01 02:00:01", "2025-09-24 00:00:02"],
        "fotos_finales": 80000, "estaciones": 276, "filas": 20000000, "porcentaje_llena": 5.8,
    }
    texto = formatear_resumen(resumen)
    assert "24.959.284" in texto and "91.989" in texto
