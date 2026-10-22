import pandas as pd

from valenbisi.limpieza import (
    REGLAS_GLOBALES,
    REGLAS_POR_TROZO,
    aplicar_reglas,
    marcar_llena,
    quitar_desactualizadas,
    quitar_duplicados,
    quitar_estacion_pruebas,
    quitar_fotos_identicas,
    quitar_hora_repetida,
    quitar_inactivas,
    quitar_sin_bicis_ni_huecos,
)


def filas(*registros) -> pd.DataFrame:
    """registros: (ts local 'YYYY-MM-DD HH:MM', estacion, activa, bicis, libres, actualizada).

    ``ts_utc`` se fabrica restando 1 h (invierno); ``actualizada=None`` deja la fecha sin leer.
    """
    df = pd.DataFrame(registros, columns=["ts", "estacion", "activa", "bicis", "libres", "fecha_actualizacion"])
    df["ts"] = pd.to_datetime(df["ts"])
    df["fecha_actualizacion"] = pd.to_datetime(df["fecha_actualizacion"])
    df.insert(1, "ts_utc", (df["ts"] - pd.Timedelta(hours=1)).dt.tz_localize("UTC"))
    df["total"] = 15
    return df


def test_quita_la_hora_repetida_sin_ts_utc():
    df = filas(("2024-10-27 01:45", 1, True, 5, 10, "2024-10-27 01:40"),
               ("2024-10-27 02:15", 1, True, 5, 10, "2024-10-27 01:40"))
    df.loc[1, "ts_utc"] = pd.NaT
    assert quitar_hora_repetida(df)["ts"].dt.hour.tolist() == [1]


def test_quita_la_estacion_299():
    df = filas(("2024-01-01 10:00", 299, True, 2, 0, "2024-01-01 09:55"),
               ("2024-01-01 10:00", 1, True, 5, 10, "2024-01-01 09:55"))
    assert quitar_estacion_pruebas(df)["estacion"].tolist() == [1]


def test_quita_registros_inactivos():
    df = filas(("2024-01-01 10:00", 1, False, 5, 10, "2024-01-01 09:55"),
               ("2024-01-01 10:00", 2, True, 5, 10, "2024-01-01 09:55"))
    assert quitar_inactivas(df)["estacion"].tolist() == [2]


def test_desactualizada_25h_se_quita_y_23h_se_conserva():
    df = filas(("2024-01-02 10:00", 1, True, 5, 10, "2024-01-01 09:00"),  # 25 h
               ("2024-01-02 10:00", 2, True, 5, 10, "2024-01-01 11:00"),  # 23 h
               ("2024-01-02 10:00", 3, True, 5, 10, "2024-01-01 10:00"),  # 24 h justas: se conserva
               ("2024-01-02 10:00", 4, True, 5, 10, None))                # fecha ilegible
    assert quitar_desactualizadas(df)["estacion"].tolist() == [2, 3]


def test_foto_identica_a_la_anterior_se_quita():
    df = filas(("2024-01-01 10:00", 1, True, 5, 10, "2024-01-01 09:55"),
               ("2024-01-01 10:00", 2, True, 3, 12, "2024-01-01 09:50"),
               # 10:15 es idéntica a 10:00 (mismos números y misma actualización)
               ("2024-01-01 10:15", 1, True, 5, 10, "2024-01-01 09:55"),
               ("2024-01-01 10:15", 2, True, 3, 12, "2024-01-01 09:50"),
               # 10:30 cambia
               ("2024-01-01 10:30", 1, True, 4, 11, "2024-01-01 10:20"),
               ("2024-01-01 10:30", 2, True, 3, 12, "2024-01-01 09:50"))
    resultado = quitar_fotos_identicas(df)
    assert resultado["ts"].dt.strftime("%H:%M").unique().tolist() == ["10:00", "10:30"]


def test_foto_con_mismos_numeros_pero_actualizacion_nueva_se_conserva():
    df = filas(("2024-01-01 04:00", 1, True, 5, 10, "2024-01-01 03:55"),
               ("2024-01-01 04:15", 1, True, 5, 10, "2024-01-01 04:10"))
    assert len(quitar_fotos_identicas(df)) == 2


def test_fotos_identicas_no_depende_del_orden_de_las_filas():
    df = filas(("2024-01-01 10:15", 2, True, 3, 12, "2024-01-01 09:50"),
               ("2024-01-01 10:00", 1, True, 5, 10, "2024-01-01 09:55"),
               ("2024-01-01 10:15", 1, True, 5, 10, "2024-01-01 09:55"),
               ("2024-01-01 10:00", 2, True, 3, 12, "2024-01-01 09:50"))
    assert quitar_fotos_identicas(df)["ts"].dt.strftime("%H:%M").unique().tolist() == ["10:00"]


def test_quita_cero_bicis_y_cero_huecos():
    df = filas(("2024-01-01 10:00", 1, True, 0, 0, "2024-01-01 09:55"),
               ("2024-01-01 10:00", 2, True, 0, 15, "2024-01-01 09:55"),
               ("2024-01-01 10:00", 3, True, 15, 0, "2024-01-01 09:55"))
    assert quitar_sin_bicis_ni_huecos(df)["estacion"].tolist() == [2, 3]


def test_quita_duplicados_de_ts_utc_y_estacion():
    df = filas(("2024-01-01 10:00", 1, True, 5, 10, "2024-01-01 09:55"),
               ("2024-01-01 10:00", 1, True, 5, 10, "2024-01-01 09:55"))
    assert len(quitar_duplicados(df)) == 1


def test_llena_solo_cuando_no_hay_libres():
    df = filas(("2024-01-01 10:00", 1, True, 15, 0, "2024-01-01 09:55"),
               ("2024-01-01 10:00", 2, True, 14, 1, "2024-01-01 09:55"),
               ("2024-01-01 10:00", 3, True, 0, 15, "2024-01-01 09:55"))
    assert marcar_llena(df)["llena"].tolist() == [True, False, False]


def test_conteos_del_resumen_cuadran_y_se_acumulan_por_trozos():
    trozo_1 = filas(("2024-01-01 10:00", 299, True, 2, 0, "2024-01-01 09:55"),
                    ("2024-01-01 10:00", 1, False, 5, 10, "2024-01-01 09:55"),
                    ("2024-01-01 10:00", 2, True, 5, 10, "2024-01-01 09:55"))
    trozo_2 = filas(("2024-01-02 10:00", 2, True, 5, 10, "2023-12-30 09:55"),  # desactualizada
                    ("2024-01-02 10:00", 3, True, 0, 0, "2024-01-02 09:55"),   # 0 y 0
                    ("2024-01-02 10:00", 4, True, 9, 6, "2024-01-02 09:55"))
    conteo: dict[str, list[int]] = {}
    limpio = pd.concat([aplicar_reglas(t, REGLAS_POR_TROZO, conteo) for t in (trozo_1, trozo_2)])
    limpio = aplicar_reglas(limpio, REGLAS_GLOBALES, conteo)

    pasos = list(conteo.values())
    assert pasos[0][0] == 6
    assert pasos[-1][1] == len(limpio) == 2
    # Encadenado: lo que sale de una regla es lo que entra en la siguiente.
    for anterior, siguiente in zip(pasos, pasos[1:]):
        assert anterior[1] == siguiente[0]
    eliminadas = {nombre: antes - despues for nombre, (antes, despues) in conteo.items()}
    assert eliminadas == {
        "Hora repetida de octubre": 0,
        "Estación de pruebas (299)": 1,
        "Registros inactivos": 1,
        "Actualización con más de 24 h": 1,
        "Foto idéntica a la anterior": 0,
        "0 bicis y 0 huecos libres": 1,
        "Duplicados (ts_utc, estación)": 0,
    }
