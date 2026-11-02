"""Script de XGBoost v2: configuraciones, prueba de humo con un Parquet sintético, reanudar, --humo y salvaguarda."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from fabrica import parquet_features_v2
from valenbisi.evaluacion import FRONTERAS_V2, limites
from valenbisi.features_v2 import FEATURES_V2_FINALES
from valenbisi.modelo import SecuenciaParquet
from valenbisi.modelo_xgboost import predecir_xgb
from valenbisi.reanudar import clave

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "xgboost_v2.py"


def cargar():
    spec = importlib.util.spec_from_file_location("xgboost_v2", SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_configuraciones_fijadas():
    m = cargar()
    assert m.PARTIDA == "X1"
    p = {n: m.parametros(n) for n in m.CONFIGURACIONES}
    assert (p["X1"]["grow_policy"], p["X1"]["max_leaves"], p["X1"]["max_depth"]) == ("lossguide", 63, 0)
    assert (p["X2"]["grow_policy"], p["X2"]["max_leaves"], p["X2"]["max_depth"]) == ("lossguide", 255, 0)
    assert (p["X3"]["grow_policy"], p["X3"]["max_depth"]) == ("depthwise", 8)
    assert all(q["eta"] == 0.05 and q["max_bin"] == 256 and q["min_child_weight"] == 50 for q in p.values())
    assert m.MAX_ARBOLES == 3000 and m.PACIENCIA == 100
    claves = {clave(FEATURES_V2_FINALES, 42, q, m.MAX_ARBOLES, m.PACIENCIA) for q in p.values()}
    assert len(claves) == len(m.CONFIGURACIONES)


def preparar(m, tmp_path, monkeypatch, contador: list, reanudar: bool = False, extra: list = ()):
    ruta = tmp_path / "features_v2.parquet"
    if not ruta.exists():
        parquet_features_v2(ruta)
    monkeypatch.setattr(m, "PARAMETROS_XGB", {**m.PARAMETROS_XGB, "min_child_weight": 1, "nthread": 2})
    monkeypatch.setattr(m, "MAX_ARBOLES", 15)
    monkeypatch.setattr(m, "PACIENCIA", 5)
    monkeypatch.setattr(m, "REPLICAS", 5)
    entrenar_real = m.entrenar_xgb

    def contar(*args, **kwargs):
        contador.append(1)
        return entrenar_real(*args, **kwargs)
    monkeypatch.setattr(m, "entrenar_xgb", contar)
    monkeypatch.setattr(sys, "argv", [
        "xgboost_v2.py", "--features", str(ruta), "--salida", str(tmp_path / "resumen.json"),
        "--predicciones", str(tmp_path / "pred"), "--modelos", str(tmp_path / "modelos"), "--minimo-libre-gb", "0",
        *(["--reanudar"] if reanudar else []), *extra])


def test_busqueda_completa_con_parquet_sintetico(tmp_path, monkeypatch):
    m = cargar()
    contador = []
    preparar(m, tmp_path, monkeypatch, contador)
    m.main()
    r = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    d = r["decisiones"]
    assert d["elegida"] in m.CONFIGURACIONES
    assert f"{d['mejor_logloss']}_s43" in r["variantes"] and f"{d['elegida']}_s43" in r["variantes"]
    assert {"X2 - X1 (todas)", "X3 - X1 (todas)"} <= set(r["comparaciones"])
    assert d["parametros_elegidos"] == m.parametros(d["elegida"])
    assert len(contador) == len(r["variantes"]) == 4 + (d["elegida"] != d["mejor_logloss"])
    # Cada modelo guardado predice exactamente lo que se midió.
    val = SecuenciaParquet(tmp_path / "features_v2.parquet", FEATURES_V2_FINALES, limites("validacion", FRONTERAS_V2))
    for nombre, v in r["variantes"].items():
        modelo = xgb.Booster(model_file=str(tmp_path / "modelos" / v["modelo"]))
        assert modelo.num_boosted_rounds() == v["arboles"] <= 15
        np.testing.assert_allclose(predecir_xgb(modelo, val), np.load(tmp_path / "pred" / f"{nombre}.npy"),
                                   atol=1e-7)

    # Una segunda ejecución con --reanudar no entrena nada y decide lo mismo.
    contador = []
    m = cargar()
    preparar(m, tmp_path, monkeypatch, contador, reanudar=True)
    m.main()
    despues = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert contador == []
    assert set(despues["reutilizadas"]) == set(r["variantes"])
    assert despues["decisiones"]["elegida"] == d["elegida"]


def test_si_la_mejor_no_gana_claro_se_repite_tambien_la_de_partida(tmp_path, monkeypatch):
    m = cargar()
    contador = []
    preparar(m, tmp_path, monkeypatch, contador)
    configuraciones = {**m.CONFIGURACIONES, "X1": {**m.CONFIGURACIONES["X1"], "eta": 0.001}}
    monkeypatch.setattr(m, "CONFIGURACIONES", configuraciones)       # X1 casi no aprende
    monkeypatch.setattr(m, "elegir_configuracion", lambda mejor, partida, diferencia, ruido: partida)
    m.main()
    r = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    d = r["decisiones"]
    assert d["mejor_logloss"] != "X1" and d["elegida"] == "X1"
    assert {"X1", "X2", "X3", f"{d['mejor_logloss']}_s43", "X1_s43"} == set(r["variantes"])
    assert d["ruido_semilla_elegida"] == pytest.approx(
        abs(r["variantes"]["X1"]["validacion"]["todas"]["logloss"]
            - r["variantes"]["X1_s43"]["validacion"]["todas"]["logloss"]))
    assert len(contador) == 5


def test_humo_mide_la_memoria_sin_escribir_nada(tmp_path, monkeypatch):
    m = cargar()
    preparar(m, tmp_path, monkeypatch, [], extra=["--humo"])
    monkeypatch.setattr(m, "HUMO_ARBOLES", 3)
    resultado = {}
    humo_real = m.humo
    monkeypatch.setattr(m, "humo", lambda *a: resultado.update(humo_real(*a)))
    m.main()
    assert not (tmp_path / "resumen.json").exists() and not (tmp_path / "pred").exists()
    assert resultado["filas_usadas"]["entrenamiento"] < resultado["filas_totales"]["entrenamiento"]
    assert resultado["gb_pico_estimado_todas"] >= resultado["gb_pico"] > 0


def test_recortar_deja_los_primeros_bloques():
    m = cargar()

    class Falsa:
        filas_bloque = [np.arange(3), np.arange(0), np.arange(2), np.arange(4)]
        _cache = None
    s = m.recortar(Falsa(), 2)
    assert [len(f) for f in s.filas_bloque] == [3, 0, 2, 0]
    assert list(s.inicio) == [0, 3, 3, 5] and list(s.fin) == [3, 3, 5, 5]


def test_se_para_si_el_parquet_tiene_datos_de_la_prueba_v2(tmp_path, monkeypatch):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    m = cargar()
    ruta = tmp_path / "f.parquet"
    df = pd.DataFrame({"slot_utc": pd.to_datetime(["2025-10-01"]).tz_localize("UTC").astype("datetime64[us, UTC]")})
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ruta)
    for extra in ([], ["--humo"]):
        monkeypatch.setattr(sys, "argv", ["x", "--features", str(ruta), "--salida", str(tmp_path / "r.json"),
                                          "--predicciones", str(tmp_path / "p"), "--modelos", str(tmp_path / "m"),
                                          "--minimo-libre-gb", "0", *extra])
        with pytest.raises(ValueError, match="prueba v2"):
            m.main()
