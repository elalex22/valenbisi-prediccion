import numpy as np
import pandas as pd

from valenbisi.evaluacion import FRONTERAS, metricas, repartir, tabla_calibracion

F1, F2 = FRONTERAS["validacion"], FRONTERAS["prueba"]


def filas(*slots: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({"slot_utc": list(slots), "estacion": 1, "llena": False, "llena_60": False})


def minutos(n: int) -> pd.Timedelta:
    return pd.Timedelta(minutes=n)


def test_fronteras_son_medianoche_en_hora_local():
    # 1-sep-2024 00:00 CEST = 31-ago 22:00 UTC; 1-mar-2025 00:00 CET = 28-feb 23:00 UTC.
    assert F1 == pd.Timestamp("2024-08-31 22:00", tz="UTC")
    assert F2 == pd.Timestamp("2025-02-28 23:00", tz="UTC")


def test_hueco_de_60_min_antes_de_cada_frontera():
    df = filas(F1 - minutos(75), F1 - minutos(60), F1 - minutos(15), F1,
               F2 - minutos(75), F2 - minutos(60), F2, F2 + minutos(60))
    conjuntos, conteo = repartir(df)

    assert conjuntos["entrenamiento"]["slot_utc"].tolist() == [F1 - minutos(75)]
    assert conjuntos["validacion"]["slot_utc"].tolist() == [F1, F2 - minutos(75)]
    assert conjuntos["prueba"]["slot_utc"].tolist() == [F2, F2 + minutos(60)]
    assert conteo["descartadas"] == {"hueco_antes_de_validacion": 2, "hueco_antes_de_prueba": 1}


def test_ninguna_etiqueta_cae_en_el_conjunto_siguiente():
    slots = pd.date_range(F1 - pd.Timedelta(days=1), F2 + pd.Timedelta(days=1), freq="15min")
    conjuntos, conteo = repartir(filas(*slots))

    assert conjuntos["entrenamiento"]["slot_utc"].max() + minutos(60) < F1
    assert conjuntos["validacion"]["slot_utc"].min() >= F1
    assert conjuntos["validacion"]["slot_utc"].max() + minutos(60) < F2
    assert conjuntos["prueba"]["slot_utc"].min() >= F2
    # Cada fila está en un solo conjunto o descartada, y el conteo cuadra.
    usados = pd.concat(conjuntos.values())["slot_utc"]
    assert not usados.duplicated().any()
    assert len(usados) + sum(conteo["descartadas"].values()) == len(slots)
    assert conteo["filas"] == {nombre: len(c) for nombre, c in conjuntos.items()}


def test_metricas_de_una_prediccion_perfecta():
    y = [True, False, False, False]
    m = metricas(y, [1.0, 0.0, 0.0, 0.0], p_constante=0.25)
    assert m == {"filas": 4, "porcentaje_positivos": 25.0, "auc_pr": 1.0, "brier": 0.0, "brier_skill": 1.0}


def test_metricas_de_la_constante_no_aportan_nada():
    y = [True, False, False, False]
    m = metricas(y, [0.25] * 4, p_constante=0.25)
    assert m["auc_pr"] == 0.25  # la proporción de positivos: lo que saca el azar
    assert m["brier"] == round((0.75**2 + 3 * 0.25**2) / 4, 5)
    assert m["brier_skill"] == 0.0


def test_calibracion_con_dos_valores_da_dos_tramos():
    y = [True, False, False, False, True, True, False, True]
    p = [0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 0.7]
    tabla = tabla_calibracion(y, p)
    assert tabla["filas"].tolist() == [4, 4]
    assert tabla["p_media"].tolist() == [0.1, 0.7]
    assert tabla["frecuencia_real"].tolist() == [0.25, 0.75]


def test_calibracion_con_un_valor_muy_frecuente_separa_el_raro():
    # Como la persistencia real: casi todas las filas comparten probabilidad.
    y = [False] * 8 + [True, True]
    p = [0.1] * 9 + [0.7]
    tabla = tabla_calibracion(y, p)
    assert tabla["filas"].tolist() == [9, 1]
    assert tabla["p_media"].tolist() == [0.1, 0.7]


def test_calibracion_por_cuantiles_reparte_las_filas():
    p = np.linspace(0, 1, 100)
    tabla = tabla_calibracion(p > 0.5, p, n_tramos=4)
    assert tabla["filas"].tolist() == [25, 25, 25, 25]
    assert tabla["p_media"].is_monotonic_increasing
    assert tabla["frecuencia_real"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_calibracion_de_una_constante_da_un_tramo():
    assert tabla_calibracion([True, False], [0.3, 0.3])["filas"].tolist() == [2]
