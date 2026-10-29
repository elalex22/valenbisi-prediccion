import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from valenbisi.evaluacion import (
    CONJUNTOS,
    FRONTERAS,
    evaluar,
    evaluar_por_tramos,
    intervalos_bootstrap,
    limites,
    metricas,
    remuestreo_por_dias,
    repartir,
    tabla_calibracion,
)

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


def test_evaluar_separa_por_llena_ahora():
    conjunto = pd.DataFrame({"llena": [True, True, False, False], "llena_60": [True, False, True, False]})
    resultado = evaluar(conjunto, {"modelo": np.array([0.9, 0.2, 0.6, 0.1])}, p_constante=0.5)

    assert set(resultado) == {"todas", "llena_ahora", "no_llena_ahora"}
    assert resultado["todas"]["modelo"]["filas"] == 4
    assert resultado["llena_ahora"]["modelo"]["filas"] == 2
    assert resultado["no_llena_ahora"]["modelo"]["auc_pr"] == 1.0


def test_evaluar_por_tramos_de_libres():
    conjunto = pd.DataFrame({"libres": [0, 0, 1, 2, 3, 3, 20],
                             "llena_60": [True, False, True, False, True, False, True]})
    p = np.array([0.7, 0.6, 0.3, 0.1, 0.2, 0.05, 0.0])
    tramos = {"0": (0, 0), "1-3": (1, 3), "4+": (4, None)}
    resultado = evaluar_por_tramos(conjunto, {"modelo": p}, p_constante=0.5, tramos=tramos)

    assert [resultado[t]["modelo"]["filas"] for t in tramos] == [2, 4, 1]
    assert resultado["1-3"]["modelo"]["porcentaje_positivos"] == 50.0
    assert resultado["1-3"]["modelo"]["auc_pr"] == 1.0


def test_limites_leen_del_parquet_las_mismas_filas_que_repartir(tmp_path):
    slots = [F1 - minutos(m) for m in (90, 60, 45, 0)] + [F1, F1 + minutos(15)] +             [F2 - minutos(m) for m in (75, 60, 15)] + [F2, F2 + minutos(600)]
    df = filas(*slots).assign(slot_utc=lambda d: d["slot_utc"].astype("datetime64[us, UTC]"))
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp_path / "f.parquet")
    conjuntos, _ = repartir(df)

    for nombre in CONJUNTOS:
        leido = pd.read_parquet(tmp_path / "f.parquet", filters=limites(nombre) or None)
        assert leido["slot_utc"].tolist() == conjuntos[nombre]["slot_utc"].tolist(), nombre


def datos_por_dias(n_dias: int = 30, filas_por_dia: int = 200, semilla: int = 0):
    """y con ~10 % de positivos; ``bueno`` ordena bien y ``malo`` es ruido."""
    rng = np.random.default_rng(semilla)
    dias = np.repeat(pd.date_range("2025-03-01", periods=n_dias).strftime("%Y-%m-%d").to_numpy(), filas_por_dia)
    y = rng.random(len(dias)) < 0.1
    bueno = np.clip(np.where(y, 0.5, 0.05) + rng.normal(0, 0.1, len(y)), 0.001, 0.999)
    malo = rng.random(len(y)) * 0.2
    return y, {"bueno": bueno, "malo": malo}, dias


def test_remuestreo_coge_dias_enteros():
    # Días de tamaños distintos y filas desordenadas.
    dias = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 2 + ["a"] * 1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        indices = remuestreo_por_dias(dias, rng)
        tamano = {"a": 4, "b": 5, "c": 2}
        cuenta = pd.Series(dias[indices]).value_counts()
        # Cada día aparece un número entero de veces completo, y se sortean 3 días.
        assert all(cuenta[d] % tamano[d] == 0 for d in cuenta.index)
        assert sum(cuenta[d] // tamano[d] for d in cuenta.index) == 3
        # Cada vez que un día entra, entran todas sus filas.
        for d in cuenta.index:
            filas = np.flatnonzero(dias == d)
            assert all((indices == f).sum() == cuenta[d] // tamano[d] for f in filas)


def test_bootstrap_es_reproducible_y_contiene_la_estimacion():
    y, predicciones, dias = datos_por_dias()
    a = intervalos_bootstrap(y, predicciones, dias, p_constante=0.1, n=50, semilla=3)
    b = intervalos_bootstrap(y, predicciones, dias, p_constante=0.1, n=50, semilla=3)
    c = intervalos_bootstrap(y, predicciones, dias, p_constante=0.1, n=50, semilla=4)
    assert a == b
    assert a != c

    esperado = metricas(y, predicciones["bueno"], p_constante=0.1)
    auc = a["bueno"]["auc_pr"]
    assert auc["estimacion"] == esperado["auc_pr"]
    assert a["bueno"]["brier_skill"]["estimacion"] == esperado["brier_skill"]
    for modelo in predicciones:
        for medida in a[modelo].values():
            assert medida["inferior"] <= medida["estimacion"] <= medida["superior"]
    assert auc["superior"] > auc["inferior"]


def test_bootstrap_de_una_diferencia_clara_no_cruza_el_cero():
    y, predicciones, dias = datos_por_dias()
    r = intervalos_bootstrap(y, predicciones, dias, p_constante=0.1, diferencias=[("bueno", "malo")], n=50)
    diferencia = r["bueno - malo"]["auc_pr"]
    assert diferencia["estimacion"] == pytest.approx(r["bueno"]["auc_pr"]["estimacion"]
                                                     - r["malo"]["auc_pr"]["estimacion"], abs=1e-4)
    assert diferencia["inferior"] > 0
