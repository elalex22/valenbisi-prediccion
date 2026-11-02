"""Script de la logística v2: prueba de humo con un Parquet sintético pequeño, reanudar y salvaguarda de la prueba v2."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fabrica import parquet_features_v2
from valenbisi.evaluacion import FRONTERAS_V2, limites
from valenbisi.features_v2 import FEATURES_V2_FINALES
from valenbisi.logistica import Logistica
from valenbisi.modelo import SecuenciaParquet
from valenbisi.reanudar import clave

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "logistica_v2.py"


def cargar():
    spec = importlib.util.spec_from_file_location("logistica_v2", SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_configuraciones_fijadas():
    m = cargar()
    assert m.PARTIDA == "R1"
    assert [m.parametros(n)["alpha"] for n in m.CONFIGURACIONES] == [1e-5, 1e-6]
    assert m.MAX_EPOCAS == 8 and m.PACIENCIA_EPOCAS == 1
    claves = {clave(FEATURES_V2_FINALES, 42, m.parametros(n), None, None) for n in m.CONFIGURACIONES}
    assert len(claves) == len(m.CONFIGURACIONES)


def preparar(m, tmp_path, monkeypatch, contador: list, reanudar: bool = False):
    ruta = tmp_path / "features_v2.parquet"
    if not ruta.exists():
        parquet_features_v2(ruta)
    monkeypatch.setattr(m, "MAX_EPOCAS", 3)
    monkeypatch.setattr(m, "REPLICAS", 5)
    ajustar_real = m.ajustar

    def contar(*args, **kwargs):
        contador.append(1)
        return ajustar_real(*args, **kwargs)
    monkeypatch.setattr(m, "ajustar", contar)
    monkeypatch.setattr(sys, "argv", [
        "logistica_v2.py", "--features", str(ruta), "--salida", str(tmp_path / "resumen.json"),
        "--predicciones", str(tmp_path / "pred"), "--modelos", str(tmp_path / "modelos"), "--minimo-libre-gb", "0",
        *(["--reanudar"] if reanudar else [])])


def test_ajuste_completo_con_parquet_sintetico(tmp_path, monkeypatch):
    m = cargar()
    contador = []
    preparar(m, tmp_path, monkeypatch, contador)
    m.main()
    r = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    d = r["decisiones"]
    assert d["elegida"] in m.CONFIGURACIONES
    assert f"{d['mejor_logloss']}_s43" in r["variantes"] and f"{d['elegida']}_s43" in r["variantes"]
    assert "R2 - R1 (todas)" in r["comparaciones"]
    assert d["parametros_elegidos"] == m.parametros(d["elegida"])
    assert len(contador) == len(r["variantes"]) == 3 + (d["elegida"] != d["mejor_logloss"])
    assert r["tramos"]["indicadores"] == sum(r["tramos"]["por_variable"].values())
    assert r["tramos"]["por_variable"]["libres"] == 14          # 0–10, 11–15, 16 o más y «falta»
    val = SecuenciaParquet(tmp_path / "features_v2.parquet", FEATURES_V2_FINALES, limites("validacion", FRONTERAS_V2))
    for nombre, v in r["variantes"].items():
        assert 1 <= v["mejor_epoca"] <= v["epocas"] <= 3
        assert v["validacion"]["todas"]["logloss"] == pytest.approx(min(v["logloss_por_epoca"]), abs=1e-6)
        # Cada modelo guardado predice exactamente lo que se midió.
        modelo = Logistica.cargar(tmp_path / "modelos" / v["modelo"])
        np.testing.assert_allclose(modelo.predecir_por_bloques(val.bloques()),
                                   np.load(tmp_path / "pred" / f"{nombre}.npy"), atol=1e-7)

    # Una segunda ejecución con --reanudar no ajusta nada y decide lo mismo.
    contador = []
    m = cargar()
    preparar(m, tmp_path, monkeypatch, contador, reanudar=True)
    m.main()
    despues = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert contador == []
    assert set(despues["reutilizadas"]) == set(r["variantes"])
    assert despues["decisiones"]["elegida"] == d["elegida"]
    for nombre in r["variantes"]:
        assert despues["variantes"][nombre]["validacion"]["todas"]["logloss"] == pytest.approx(
            r["variantes"][nombre]["validacion"]["todas"]["logloss"], abs=1e-6)


def test_si_la_mejor_no_gana_claro_se_repite_tambien_la_de_partida(tmp_path, monkeypatch):
    m = cargar()
    contador = []
    preparar(m, tmp_path, monkeypatch, contador)
    monkeypatch.setattr(m, "CONFIGURACIONES", {"R1": {"alpha": 1e-1}, "R2": {"alpha": 1e-6}})  # R1 mucho peor
    monkeypatch.setattr(m, "elegir_configuracion", lambda mejor, partida, diferencia, ruido: partida)
    m.main()
    r = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    d = r["decisiones"]
    assert d["mejor_logloss"] == "R2" and d["elegida"] == "R1"
    assert {"R1", "R2", "R2_s43", "R1_s43"} == set(r["variantes"])
    assert d["ruido_semilla_elegida"] == pytest.approx(
        abs(r["variantes"]["R1"]["validacion"]["todas"]["logloss"]
            - r["variantes"]["R1_s43"]["validacion"]["todas"]["logloss"]))
    assert len(contador) == 4


def test_se_para_si_el_parquet_tiene_datos_de_la_prueba_v2(tmp_path, monkeypatch):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    m = cargar()
    ruta = tmp_path / "f.parquet"
    df = pd.DataFrame({"slot_utc": pd.to_datetime(["2025-10-01"]).tz_localize("UTC").astype("datetime64[us, UTC]")})
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ruta)
    monkeypatch.setattr(sys, "argv", ["x", "--features", str(ruta), "--salida", str(tmp_path / "r.json"),
                                      "--predicciones", str(tmp_path / "p"), "--modelos", str(tmp_path / "m"),
                                      "--minimo-libre-gb", "0"])
    with pytest.raises(ValueError, match="prueba v2"):
        m.main()
