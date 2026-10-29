"""Prueba de humo de scripts/evaluacion_final.py con un Parquet sintético pequeño.

La evaluación real solo se ejecuta una vez, así que los errores de código tienen
que salir aquí, no a mitad de la prueba de verdad.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from valenbisi.evaluacion import evaluar, limites
from valenbisi.modelo import FEATURES_MODELO, dataset, entrenar_fijo, matriz
from valenbisi.referencias import ajustar_frecuencia, ajustar_persistencia, predecir_frecuencia, predecir_persistencia

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluacion_final.py"
ARBOLES = 5


def cargar_script():
    spec = importlib.util.spec_from_file_location("evaluacion_final", SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def parquet_sintetico(ruta: Path) -> None:
    """Dos fotos al día (8 y 18 h locales) de 4 estaciones, de 2023 a jun-2025; la 5 solo en prueba."""
    rng = np.random.default_rng(0)
    dias = pd.date_range("2023-01-01", "2025-06-30", freq="D")
    locales = [d + pd.Timedelta(hours=h) for d in dias for h in (8, 18)]
    slots = pd.DatetimeIndex(locales).tz_localize("Europe/Madrid").tz_convert("UTC")
    filas = []
    for estacion in range(1, 6):
        s = slots if estacion < 5 else slots[slots >= pd.Timestamp("2025-03-01", tz="UTC")]
        filas.append(pd.DataFrame({"slot_utc": s, "estacion": estacion}))
    df = pd.concat(filas, ignore_index=True)
    n = len(df)
    libres = rng.integers(0, 10, n)
    df = df.assign(
        llena=libres == 0, libres=libres.astype("int16"), bicis=rng.integers(0, 20, n).astype("int16"),
        total=np.full(n, 20, dtype="int16"), ocupacion=rng.random(n).astype("float32"),
        delta_libres_15=rng.normal(0, 1, n).astype("float32"), delta_libres_30=rng.normal(0, 1, n).astype("float32"),
        delta_libres_60=rng.normal(0, 1, n).astype("float32"),
        lat=rng.random(n).astype("float32"), lon=rng.random(n).astype("float32"),
        minuto_del_dia=rng.integers(0, 1440, n).astype("int16"), dia_semana=rng.integers(0, 7, n).astype("int8"),
        mes=rng.integers(1, 13, n).astype("int8"), festivo=rng.random(n) < 0.05,
        llena_60=rng.random(n) < np.where(libres == 0, 0.6, 0.3 / (1 + libres)),
    )
    df["estacion"] = df["estacion"].astype("int16")
    df["slot_utc"] = df["slot_utc"].astype("datetime64[us, UTC]")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ruta)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    features = tmp_path / "valenbisi_features.parquet"
    parquet_sintetico(features)
    modulo = cargar_script()
    filas = {c: pq.read_table(features, columns=["slot_utc"], filters=limites(c)).num_rows
             for c in ("entrenamiento", "validacion", "prueba")}
    monkeypatch.setattr(modulo, "FILAS_ESPERADAS", filas)
    monkeypatch.setattr(modulo, "ARBOLES_FINALES", ARBOLES)
    monkeypatch.setattr(modulo, "REPLICAS_BOOTSTRAP", 3)
    parametros = {**modulo.PARAMETROS, "min_data_in_leaf": 20, "num_threads": 2}
    monkeypatch.setattr(modulo, "PARAMETROS", parametros)

    # Modelo "oficial" de juguete: solo entrenamiento, las 15 features y ARBOLES árboles.
    tabla = pq.read_table(features, columns=[*FEATURES_MODELO, "llena_60"], filters=limites("entrenamiento"))
    oficial = entrenar_fijo(dataset(matriz(tabla), tabla.column("llena_60").to_numpy()), ARBOLES, parametros)
    oficial.save_model(tmp_path / "lightgbm.txt")

    # Las cifras «publicadas» de las referencias en prueba, calculadas como lo hizo evaluar_referencias.py.
    columnas = ["slot_utc", "estacion", "llena", "llena_60"]
    entrenamiento = pd.read_parquet(features, columns=columnas, filters=limites("entrenamiento"))
    prueba = pd.read_parquet(features, columns=columnas, filters=limites("prueba"))
    predicciones = {"persistencia": predecir_persistencia(ajustar_persistencia(entrenamiento), prueba),
                    "frecuencia": predecir_frecuencia(ajustar_frecuencia(entrenamiento), prueba)[0]}
    publicadas = evaluar(prueba, predicciones, float(entrenamiento["llena_60"].mean()))
    (tmp_path / "resumen_referencias.json").write_text(json.dumps({"prueba": publicadas}), encoding="utf-8")

    argumentos = ["evaluacion_final.py", "--features", str(features),
                  "--modelo-oficial", str(tmp_path / "lightgbm.txt"), "--modelo-final", str(tmp_path / "lightgbm_final.txt"), "--figuras", str(tmp_path / "figuras"),
                  "--minimo-libre-gb", "0"]
    return modulo, tmp_path, argumentos


def test_sin_confirmar_no_lee_nada(entorno, monkeypatch):
    modulo, tmp_path, argumentos = entorno
    monkeypatch.setattr(sys, "argv", argumentos)
    with pytest.raises(SystemExit):
        modulo.main()
    assert not (tmp_path / "lightgbm_final.txt").exists()
    assert not (tmp_path / "resumen_prueba.json").exists()


def test_evaluacion_completa_de_principio_a_fin(entorno, monkeypatch):
    modulo, tmp_path, argumentos = entorno
    monkeypatch.setattr(sys, "argv", [*argumentos, "--confirmar-prueba"])
    modulo.main()

    resumen = json.loads((tmp_path / "resumen_prueba.json").read_text(encoding="utf-8"))
    assert set(resumen["prueba"]) == {"principal", "secundario"}
    for escenario in resumen["prueba"].values():
        assert {"todas", "llena_ahora", "no_llena_ahora", "0", "1", "2-3", "4-6", "7+"} <= set(escenario)
        assert set(escenario["todas"]) == set(modulo.MODELOS)
    # El modelo final se reentrena con más datos: sus predicciones no son las del oficial.
    assert (resumen["prueba"]["principal"]["todas"]["lightgbm"]["brier"]
            != resumen["prueba"]["secundario"]["todas"]["lightgbm"]["brier"])
    assert {"lightgbm - libres", "lightgbm - lightgbm_solo_entrenamiento"} <= set(resumen["intervalos_95"])
    assert [m["mes"] for m in resumen["calibracion_por_mes"]] == ["2025-03", "2025-04", "2025-05", "2025-06"]
    assert resumen["estaciones_nuevas"]["estaciones"] == 1
    assert resumen["decisiones"]["arboles_finales"] == ARBOLES
    assert (tmp_path / "lightgbm_final.txt").exists()
    assert {p.name for p in (tmp_path / "figuras").iterdir()} >= {"calibracion_prueba.png", "curvas_pr_prueba.png",
                                                                  "mensual_prueba.png"}


def test_para_si_las_referencias_no_cuadran_con_lo_publicado(entorno, monkeypatch):
    modulo, tmp_path, argumentos = entorno
    ruta = tmp_path / "resumen_referencias.json"
    publicadas = json.loads(ruta.read_text(encoding="utf-8"))
    publicadas["prueba"]["todas"]["persistencia"]["auc_pr"] += 0.01
    ruta.write_text(json.dumps(publicadas), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [*argumentos, "--confirmar-prueba"])

    with pytest.raises(ValueError, match="no cuadra"):
        modulo.main()
    assert not (tmp_path / "resumen_prueba.json").exists()
