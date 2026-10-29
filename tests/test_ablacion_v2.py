"""Reglas de la ablación v2 y prueba de humo del script con un Parquet sintético pequeño."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from valenbisi.features_v2 import FEATURES_RETARDOS, FEATURES_V2
from valenbisi.modelo import FEATURES_MODELO

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ablacion_v2.py"


def cargar():
    spec = importlib.util.spec_from_file_location("ablacion_v2", SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def dif(estimacion: float, inferior: float, superior: float) -> dict:
    return {"estimacion": estimacion, "inferior": inferior, "superior": superior}


def test_acepta_solo_mejoras_claras_y_mayores_que_el_ruido():
    m = cargar()
    assert m.acepta(dif(-0.0005, -0.0008, -0.0002), ruido=0.0001)
    assert not m.acepta(dif(-0.0005, -0.0008, 0.0001), ruido=0.0001)     # el intervalo cruza el 0
    assert not m.acepta(dif(-0.00015, -0.0002, -0.0001), ruido=0.0001)   # mejora menor que 2 × ruido
    assert not m.acepta(dif(0.0003, 0.0001, 0.0005), ruido=0.0)          # empeora


def test_eleccion_de_la_lluvia():
    m = cargar()
    assert m.elegir_lluvia(True, True, dif(-0.0002, -0.0003, -0.0001)) == "meteo_cantidad"
    assert m.elegir_lluvia(True, True, dif(-0.0002, -0.0004, 0.0001)) == "meteo_binaria"   # empate: la sencilla
    assert m.elegir_lluvia(True, True, dif(0.0002, 0.0001, 0.0003)) == "meteo_binaria"
    assert m.elegir_lluvia(True, False, None) == "meteo_cantidad"
    assert m.elegir_lluvia(False, True, None) == "meteo_binaria"
    assert m.elegir_lluvia(False, False, None) is None


def test_etapa_2_y_columnas():
    m = cargar()
    assert m.se_queda(dif(0.0003, 0.0001, 0.0005))          # quitarlo empeora: se queda
    assert not m.se_queda(dif(0.0001, -0.0001, 0.0003))     # no está claro: fuera
    cols = m.columnas(["vecinas", "retardos"])
    assert cols == [c for c in FEATURES_V2 if c in set(FEATURES_MODELO) | set(cols)]
    assert set(FEATURES_RETARDOS) <= set(cols) and "temp" not in cols and "tmed_ayer" not in cols
    assert m.columnas([]) == FEATURES_MODELO


def parquet_sintetico(ruta: Path) -> None:
    """Dic-2024 a may-2025 (entrenamiento hasta feb, validación desde mar), 4 slots al día y 4 estaciones."""
    rng = np.random.default_rng(0)
    slots = pd.date_range("2024-12-01", "2025-05-31 23:00", freq="6h", tz="Europe/Madrid").tz_convert("UTC")
    df = pd.DataFrame([(s, e) for s in slots for e in range(1, 5)], columns=["slot_utc", "estacion"])
    n = len(df)
    for c in FEATURES_V2:
        df[c] = rng.random(n).astype("float32")
    df["estacion"] = df["estacion"].astype("int16")
    df["libres"] = rng.integers(0, 10, n).astype("int16")
    df["llena"] = df["libres"] == 0
    hueco = rng.random(n) < 0.1
    df.loc[hueco, FEATURES_RETARDOS] = np.nan
    df["llena_60"] = rng.random(n) < np.where(df["libres"] == 0, 0.6, 0.15) + 0.2 * df["vec_ocupacion"]
    df["slot_utc"] = df["slot_utc"].astype("datetime64[us, UTC]")
    pq.write_table(pa.Table.from_pandas(df[["slot_utc", *FEATURES_V2, "llena_60"]], preserve_index=False), ruta,
                   row_group_size=700)


def test_ablacion_completa_con_parquet_sintetico(tmp_path, monkeypatch):
    m = cargar()
    ruta = tmp_path / "features_v2.parquet"
    parquet_sintetico(ruta)
    monkeypatch.setattr(m, "PARAMETROS", {**m.PARAMETROS, "min_data_in_leaf": 20, "num_threads": 2})
    monkeypatch.setattr(m, "MAX_ARBOLES", 20)
    monkeypatch.setattr(m, "REPLICAS", 5)
    monkeypatch.setattr(m, "acepta", lambda diferencia, ruido: True)   # para recorrer también la etapa 2
    salida, carpeta = tmp_path / "resumen.json", tmp_path / "pred"
    monkeypatch.setattr(sys, "argv", ["ablacion_v2.py", "--features", str(ruta), "--salida", str(salida),
                                      "--predicciones", str(carpeta), "--minimo-libre-gb", "0"])
    m.main()

    r = json.loads(salida.read_text(encoding="utf-8"))
    variantes = set(r["variantes"])
    assert {"A0", "A0b", "A1", "A2", "A3", "A4", "A4_sin_7d", "A4_sin_retardos", "A5", "A5b"} <= variantes
    d = r["decisiones"]
    assert d["lluvia"] in ("meteo_cantidad", "meteo_binaria")
    assert all(f"A5_sin_{g}" in variantes for g in d["etapa_2_candidatos"])
    assert set(d["grupos_finales"]) <= set(d["etapa_2_candidatos"])
    assert set(FEATURES_MODELO) <= set(d["features_finales"])
    # Medidas por subgrupo, con las filas sin retardo aparte.
    a0 = r["variantes"]["A0"]["validacion"]
    assert {"todas", "con_retardos", "sin_retardos", "no_llena_ahora", "libres_1_6"} <= set(a0)
    assert a0["con_retardos"]["filas"] + a0["sin_retardos"]["filas"] == a0["todas"]["filas"]
    # La simulación vacía los retardos: sus predicciones cambian respecto a A4.
    a4 = np.load(carpeta / "A4.npy")
    assert a4.dtype == np.float32 and len(a4) == r["filas"]["validacion"]
    assert not np.array_equal(a4, np.load(carpeta / "A4_sin_retardos.npy"))
    assert "A4 - A0 (con_retardos)" in r["comparaciones"]


def test_se_para_si_el_parquet_tiene_datos_de_la_prueba_v2(tmp_path, monkeypatch):
    m = cargar()
    ruta = tmp_path / "f.parquet"
    df = pd.DataFrame({"slot_utc": pd.to_datetime(["2025-10-01"]).tz_localize("UTC").astype("datetime64[us, UTC]")})
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ruta)
    monkeypatch.setattr(sys, "argv", ["ablacion_v2.py", "--features", str(ruta), "--salida", str(tmp_path / "r.json"),
                                      "--predicciones", str(tmp_path / "p"), "--minimo-libre-gb", "0"])
    with pytest.raises(ValueError, match="prueba v2"):
        m.main()


def test_clave_exige_mismas_columnas_en_orden_semilla_parametros_y_parada():
    m = cargar()
    base = m.clave(["a", "b"], 42, {"lr": 0.05}, 3000, 100)
    assert base == m.clave(["a", "b"], 42, {"lr": 0.05}, 3000, 100)
    assert base != m.clave(["b", "a"], 42, {"lr": 0.05}, 3000, 100)
    assert base != m.clave(["a", "b"], 43, {"lr": 0.05}, 3000, 100)
    assert base != m.clave(["a", "b"], 42, {"lr": 0.1}, 3000, 100)
    assert base != m.clave(["a", "b"], 42, {"lr": 0.05}, 2000, 100)
    assert base != m.clave(["a", "b"], 42, {"lr": 0.05}, 3000, 50)


def test_indice_de_previas_marca_la_parada_asumida_y_salta_simulaciones():
    m = cargar()
    previo = {"parametros": {"lr": 0.05}, "variantes": {
        "A0": {"features": ["a"], "semilla": 42},                                    # resumen antiguo
        "A4": {"features": ["a", "r"], "semilla": 42, "parametros": {"lr": 0.05}, "max_arboles": 3000,
               "paciencia": 100},
        "A4_sin_7d": {"simulacion": "sin_7d", "de": "A4"},
    }}
    indice = m.indice_previas(previo)
    assert indice[m.clave(["a"], 42, {"lr": 0.05}, 3000, 100)] == ("A0", True)
    assert indice[m.clave(["a", "r"], 42, {"lr": 0.05}, 3000, 100)] == ("A4", False)
    assert len(indice) == 2


def preparar(m, tmp_path, monkeypatch, reanudar: bool, contador: list):
    ruta = tmp_path / "features_v2.parquet"
    if not ruta.exists():
        parquet_sintetico(ruta)
    monkeypatch.setattr(m, "PARAMETROS", {**m.PARAMETROS, "min_data_in_leaf": 20, "num_threads": 2})
    monkeypatch.setattr(m, "MAX_ARBOLES", 20)
    monkeypatch.setattr(m, "REPLICAS", 5)
    monkeypatch.setattr(m, "acepta", lambda diferencia, ruido: True)
    entrenar_real = m.entrenar

    def contar(*args, **kwargs):
        contador.append(1)
        return entrenar_real(*args, **kwargs)
    monkeypatch.setattr(m, "entrenar", contar)
    argv = ["ablacion_v2.py", "--features", str(ruta), "--salida", str(tmp_path / "resumen.json"),
            "--predicciones", str(tmp_path / "pred"), "--minimo-libre-gb", "0", *(["--reanudar"] if reanudar else [])]
    monkeypatch.setattr(sys, "argv", argv)


def test_reanudar_reutiliza_todo_lo_identico_y_da_las_mismas_decisiones(tmp_path, monkeypatch):
    primera, segunda = [], []
    m = cargar()
    preparar(m, tmp_path, monkeypatch, reanudar=False, contador=primera)
    m.main()
    antes = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert primera and antes["reutilizadas"] == {}

    m = cargar()
    preparar(m, tmp_path, monkeypatch, reanudar=True, contador=segunda)
    m.main()
    despues = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))

    assert segunda == []                                              # no se entrena nada
    entrenadas = {n for n, v in despues["variantes"].items() if "features" in v}
    assert set(despues["reutilizadas"]) == entrenadas
    assert all(not nota["parada_asumida"] for nota in despues["reutilizadas"].values())
    assert despues["decisiones"]["grupos_finales"] == antes["decisiones"]["grupos_finales"]
    # «A5 sin un grupo» tiene las mismas columnas y semilla que A3 o A4: se reutiliza de ellas.
    candidatos = despues["decisiones"]["etapa_2_candidatos"]
    origenes = {despues["reutilizadas"][f"A5_sin_{g}"]["reutilizada_de"] for g in candidatos}
    assert origenes <= {"A1", "A2", "A3", "A4"}
    for nombre in entrenadas:
        assert despues["variantes"][nombre]["validacion"]["todas"]["logloss"] == pytest.approx(
            antes["variantes"][nombre]["validacion"]["todas"]["logloss"], abs=1e-6)


def test_reanudar_vuelve_a_entrenar_si_cambian_los_parametros_o_falta_la_prediccion(tmp_path, monkeypatch):
    m = cargar()
    preparar(m, tmp_path, monkeypatch, reanudar=False, contador=[])
    m.main()

    # Falta la predicción de A3: se reentrena solo esa (y las que dependan de columnas nuevas).
    (tmp_path / "pred" / "A3.npy").unlink()
    contador = []
    m = cargar()
    preparar(m, tmp_path, monkeypatch, reanudar=True, contador=contador)
    m.main()
    r = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert "A3" not in r["reutilizadas"] and "A0" in r["reutilizadas"]
    assert len(contador) >= 1

    # Otros parámetros: no se reutiliza nada.
    contador = []
    m = cargar()
    preparar(m, tmp_path, monkeypatch, reanudar=True, contador=contador)
    monkeypatch.setattr(m, "PARAMETROS", {**m.PARAMETROS, "learning_rate": 0.1})
    m.main()
    r = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert r["reutilizadas"] == {}
    assert len(contador) == sum(1 for v in r["variantes"].values() if "features" in v)


def test_reanudar_un_resumen_sin_parada_registrada_lo_indica(tmp_path, monkeypatch):
    m = cargar()
    preparar(m, tmp_path, monkeypatch, reanudar=False, contador=[])
    m.main()
    # Se imita el primer resumen real: sin parámetros ni parada por variante.
    ruta = tmp_path / "resumen.json"
    r = json.loads(ruta.read_text(encoding="utf-8"))
    for v in r["variantes"].values():
        for k in ("parametros", "max_arboles", "paciencia"):
            v.pop(k, None)
    ruta.write_text(json.dumps(r), encoding="utf-8")

    contador = []
    m = cargar()
    preparar(m, tmp_path, monkeypatch, reanudar=True, contador=contador)
    monkeypatch.setattr(m, "PARADA_SIN_REGISTRAR", {"max_arboles": 20, "paciencia": m.PACIENCIA})
    m.main()
    despues = json.loads(ruta.read_text(encoding="utf-8"))
    assert contador == []
    assert all(nota["parada_asumida"] for nota in despues["reutilizadas"].values())
