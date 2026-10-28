import numpy as np
import pandas as pd
import pytest

from valenbisi.referencias import (
    ajustar_frecuencia,
    ajustar_libres,
    ajustar_persistencia,
    calendario,
    predecir_frecuencia,
    predecir_libres,
    predecir_persistencia,
)


def filas(*registros) -> pd.DataFrame:
    """registros: (hora local 'YYYY-MM-DD HH:MM', estacion, llena, llena_60)."""
    df = pd.DataFrame(registros, columns=["local", "estacion", "llena", "llena_60"])
    slot = pd.to_datetime(df.pop("local")).dt.tz_localize("Europe/Madrid").dt.tz_convert("UTC")
    return df.assign(slot_utc=slot)


# Lunes 15-ene-2024 y sábado 20-ene-2024.
LUNES, SABADO = "2024-01-15", "2024-01-20"


def test_persistencia_es_la_proporcion_de_entrenamiento():
    entrenamiento = filas(
        (f"{LUNES} 10:00", 1, True, True), (f"{LUNES} 10:15", 1, True, False),
        (f"{LUNES} 10:00", 2, False, False), (f"{LUNES} 10:15", 2, False, False),
        (f"{LUNES} 10:30", 2, False, False), (f"{LUNES} 10:45", 2, False, True),
    )
    modelo = ajustar_persistencia(entrenamiento)
    assert modelo == {True: 0.5, False: 0.25}

    prueba = filas((f"{SABADO} 08:00", 7, True, False), (f"{SABADO} 08:00", 8, False, True))
    assert predecir_persistencia(modelo, prueba).tolist() == [0.5, 0.25]


def test_calendario_usa_la_hora_local():
    df = filas((f"{LUNES} 00:30", 1, False, False), (f"{SABADO} 23:45", 1, False, False),
               ("2024-07-14 09:00", 1, False, False))  # domingo, horario de verano
    cal = calendario(df)
    assert cal["hora"].tolist() == [0, 23, 9]
    assert cal["fin_de_semana"].tolist() == [False, True, True]


def test_frecuencia_por_estacion_hora_y_tipo_de_dia():
    entrenamiento = filas(
        (f"{LUNES} 08:00", 1, False, True), (f"{LUNES} 08:30", 1, False, False),   # lunes 8 h: 1/2
        (f"{SABADO} 08:15", 1, False, False), (f"{SABADO} 08:45", 1, False, False),  # sábado 8 h: 0/2
        (f"{LUNES} 09:00", 1, False, True), (f"{LUNES} 09:15", 1, False, True),    # lunes 9 h: 2/2
    )
    modelo = ajustar_frecuencia(entrenamiento, min_filas=1)

    prueba = filas(("2024-03-04 08:45", 1, True, False),   # otro lunes, 8 h
                   ("2024-03-09 08:00", 1, False, False),  # otro sábado, 8 h
                   ("2024-03-05 09:30", 1, False, False))  # martes, 9 h
    p, respaldo = predecir_frecuencia(modelo, prueba)
    assert p.tolist() == [0.5, 0.0, 1.0]
    assert not respaldo.any()


def test_celda_con_pocas_filas_usa_el_respaldo():
    entrenamiento = filas(
        (f"{LUNES} 08:00", 1, False, True), (f"{LUNES} 08:15", 1, False, True),
        (f"{LUNES} 08:30", 1, False, True),                                       # estación 1: 3 filas
        (f"{LUNES} 08:00", 2, False, False),                                      # estación 2: 1 fila
    )
    modelo = ajustar_frecuencia(entrenamiento, min_filas=2)

    prueba = filas((f"{LUNES} 08:00", 1, False, False), (f"{LUNES} 08:00", 2, False, False),
                   (f"{LUNES} 08:00", 9, False, False))  # estación que no estaba en entrenamiento
    p, respaldo = predecir_frecuencia(modelo, prueba)
    # El respaldo es la proporción de lunes a las 8 h en todas las estaciones: 3/4.
    assert p.tolist() == [1.0, 0.75, 0.75]
    assert respaldo.tolist() == [False, True, True]


def test_las_predicciones_no_usan_la_etiqueta():
    entrenamiento = filas((f"{LUNES} 08:00", 1, True, True), (f"{LUNES} 08:15", 1, False, False))
    persistencia, frecuencia = ajustar_persistencia(entrenamiento), ajustar_frecuencia(entrenamiento, min_filas=1)
    prueba = filas((f"{LUNES} 08:00", 1, True, False), (f"{LUNES} 08:00", 1, False, True))
    invertida = prueba.assign(llena_60=~prueba["llena_60"])

    assert np.array_equal(predecir_persistencia(persistencia, prueba), predecir_persistencia(persistencia, invertida))
    assert np.array_equal(predecir_frecuencia(frecuencia, prueba)[0], predecir_frecuencia(frecuencia, invertida)[0])
    assert predecir_frecuencia(frecuencia, prueba)[0].tolist() == pytest.approx([0.5, 0.5])


def con_libres(*registros) -> pd.DataFrame:
    """registros: (libres, llena_60). La hora y la estación no importan; llena = libres == 0."""
    libres = [r[0] for r in registros]
    return filas(*((f"{LUNES} 08:00", 1, l == 0, y) for l, y in registros)).assign(libres=libres)


def test_libres_es_la_proporcion_por_valor():
    entrenamiento = con_libres((0, True), (0, True), (0, False), (1, True), (1, False), (2, False), (2, False))
    modelo = ajustar_libres(entrenamiento, min_filas=1)
    assert modelo.to_dict() == pytest.approx({0: 2 / 3, 1: 0.5, 2: 0.0})
    assert predecir_libres(modelo, con_libres((2, True), (0, False), (1, False))).tolist() == pytest.approx(
        [0.0, 2 / 3, 0.5])


def test_libres_sin_celda_usa_la_celda_inferior():
    entrenamiento = con_libres((1, True), (1, False), (3, False), (3, True), (3, True), (3, True), (5, True))
    modelo = ajustar_libres(entrenamiento, min_filas=2)   # la celda 5 tiene una sola fila
    assert modelo.index.tolist() == [1, 3]
    prueba = con_libres((0, False), (2, False), (5, False), (40, False))
    # 0 es menor que todas: la más baja (1). 2 → 1. 5 (pocas filas) y 40 (no vista) → 3.
    assert predecir_libres(modelo, prueba).tolist() == pytest.approx([0.5, 0.5, 0.75, 0.75])


def test_libres_no_usa_la_etiqueta_y_en_llenas_es_la_persistencia():
    entrenamiento = con_libres((0, True), (0, False), (0, True), (0, True), (4, False), (4, True), (7, False))
    libres, persistencia = ajustar_libres(entrenamiento, min_filas=1), ajustar_persistencia(entrenamiento)
    prueba = con_libres((0, True), (0, False), (4, True), (7, False))
    invertida = prueba.assign(llena_60=~prueba["llena_60"])
    assert np.array_equal(predecir_libres(libres, prueba), predecir_libres(libres, invertida))

    llenas = prueba["llena"].to_numpy()
    assert predecir_libres(libres, prueba)[llenas] == pytest.approx(predecir_persistencia(persistencia, prueba)[llenas])
