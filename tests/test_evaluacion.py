import pandas as pd

from valenbisi.evaluacion import FRONTERAS, repartir

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
