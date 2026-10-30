"""Búsqueda de hiperparámetros de LightGBM v2: rejilla y prueba de humo con un Parquet sintético pequeño."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from fabrica import parquet_features_v2
from valenbisi.features_v2 import FEATURES_V2_FINALES
from valenbisi.modelo import PARAMETROS
from valenbisi.reanudar import clave

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hiperparametros_lightgbm_v2.py"


def cargar():
    spec = importlib.util.spec_from_file_location("hiperparametros_lightgbm_v2", SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_rejilla_fijada():
    m = cargar()
    assert m.parametros("L0") == PARAMETROS              # L0 = A5 de la ablación, para reutilizarlo
    assert [m.parametros(n)["num_leaves"] for n in m.CONFIGURACIONES] == [63, 31, 127, 255, 127, 255]
    assert m.parametros("L5")["min_data_in_leaf"] == 5000 and m.parametros("L5")["lambda_l2"] == 10.0
    claves = {clave(FEATURES_V2_FINALES, 42, m.parametros(n), m.MAX_ARBOLES, m.PACIENCIA) for n in m.CONFIGURACIONES}
    assert len(claves) == len(m.CONFIGURACIONES)


def preparar(m, tmp_path, monkeypatch, contador: list, ablacion: Path | None = None, reanudar: bool = False):
    ruta = tmp_path / "features_v2.parquet"
    if not ruta.exists():
        parquet_features_v2(ruta)
    pequenos = {"min_data_in_leaf": 20, "num_threads": 2}
    monkeypatch.setattr(m, "PARAMETROS", {**m.PARAMETROS, **pequenos})
    monkeypatch.setattr(m, "CONFIGURACIONES", {n: {**c, **({"min_data_in_leaf": 40} if "min_data_in_leaf" in c else {})}
                                               for n, c in m.CONFIGURACIONES.items()})
    monkeypatch.setattr(m, "MAX_ARBOLES", 15)
    monkeypatch.setattr(m, "REPLICAS", 5)
    entrenar_real = m.entrenar

    def contar(*args, **kwargs):
        contador.append(1)
        return entrenar_real(*args, **kwargs)
    monkeypatch.setattr(m, "entrenar", contar)
    abl = ablacion or tmp_path / "no_hay_ablacion.json"
    monkeypatch.setattr(sys, "argv", [
        "hiperparametros_lightgbm_v2.py", "--features", str(ruta), "--salida", str(tmp_path / "resumen.json"),
        "--predicciones", str(tmp_path / "pred"), "--modelos", str(tmp_path / "modelos"),
        "--ablacion", str(abl), "--ablacion-predicciones", str(abl.parent / "abl_pred"),
        "--ablacion-modelos", str(abl.parent / "abl_modelos"), "--minimo-libre-gb", "0",
        *(["--reanudar"] if reanudar else [])])


def test_busqueda_completa_con_parquet_sintetico(tmp_path, monkeypatch):
    m = cargar()
    contador = []
    preparar(m, tmp_path, monkeypatch, contador)
    m.main()
    r = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    d = r["decisiones"]
    assert {*m.CONFIGURACIONES, "L0_s43"} <= set(r["variantes"])
    assert d["elegida"] in m.CONFIGURACIONES and d["mejor_logloss"] in m.CONFIGURACIONES
    assert d["parametros_elegidos"] == m.parametros(d["elegida"])
    assert all(f"{n} - L0 (todas)" in r["comparaciones"] for n in m.CONFIGURACIONES if n != "L0")
    # L0 siempre tiene su semilla 43 (A5b); si gana otra configuración, también se repite.
    assert f"{d['elegida']}_s43" in r["variantes"]
    assert len(r["variantes"]) == len(m.CONFIGURACIONES) + 1 + (d["elegida"] != "L0")
    assert len(contador) == len(r["variantes"])
    # Cada modelo guardado predice exactamente lo que se midió.
    import lightgbm as lgb
    from valenbisi.modelo import SecuenciaParquet, predecir_por_lotes
    from valenbisi.evaluacion import FRONTERAS_V2, limites
    val = SecuenciaParquet(tmp_path / "features_v2.parquet", FEATURES_V2_FINALES, limites("validacion", FRONTERAS_V2))
    for nombre, v in r["variantes"].items():
        modelo = lgb.Booster(model_file=str(tmp_path / "modelos" / v["modelo"]))
        assert modelo.num_trees() == v["arboles"]
        np.testing.assert_allclose(predecir_por_lotes(modelo, val), np.load(tmp_path / "pred" / f"{nombre}.npy"),
                                   atol=1e-7)


def test_si_gana_otra_configuracion_se_repite_con_la_semilla_43(tmp_path, monkeypatch):
    m = cargar()
    contador = []
    preparar(m, tmp_path, monkeypatch, contador)
    monkeypatch.setattr(m, "elegir_configuracion", lambda mejor, partida, diferencia, ruido: "L3")
    m.main()
    r = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    d = r["decisiones"]
    assert d["elegida"] == "L3" and r["variantes"]["L3_s43"]["semilla"] == 43
    assert d["ruido_semilla_elegida"] == pytest.approx(
        abs(r["variantes"]["L3"]["validacion"]["todas"]["logloss"]
            - r["variantes"]["L3_s43"]["validacion"]["todas"]["logloss"]))
    assert len(contador) == len(m.CONFIGURACIONES) + 2


def test_reutiliza_a5_y_a5b_de_la_ablacion_y_reanuda(tmp_path, monkeypatch):
    m = cargar()
    preparar(m, tmp_path, monkeypatch, [])
    m.main()
    # Se fabrica una «ablación» cuyo A5 y A5b son L0 y L0_s43 de esta ejecución.
    previo = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    (tmp_path / "abl_pred").mkdir()
    (tmp_path / "abl_modelos").mkdir()
    variantes = {}
    for origen, destino in (("L0", "A5"), ("L0_s43", "A5b")):
        v = {k: x for k, x in previo["variantes"][origen].items() if k != "modelo"}
        variantes[destino] = {**v, "grupos": ["vecinas", "retardos"]}
        shutil.copyfile(tmp_path / "pred" / f"{origen}.npy", tmp_path / "abl_pred" / f"{destino}.npy")
        shutil.copyfile(tmp_path / "modelos" / f"{origen}.txt", tmp_path / "abl_modelos" / f"{destino}.txt")
    ablacion = tmp_path / "ablacion.json"
    ablacion.write_text(json.dumps({"variantes": variantes}), encoding="utf-8")
    shutil.rmtree(tmp_path / "pred")
    shutil.rmtree(tmp_path / "modelos")
    (tmp_path / "resumen.json").unlink()

    contador = []
    m = cargar()
    preparar(m, tmp_path, monkeypatch, contador, ablacion=ablacion)
    m.main()
    r = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert r["reutilizadas"]["L0"] == {"resumen": "ablacion.json", "variante": "A5"}
    assert r["reutilizadas"]["L0_s43"] == {"resumen": "ablacion.json", "variante": "A5b"}
    assert (tmp_path / "modelos" / "L0.txt").exists()            # el modelo de A5 se copia
    assert len(contador) == len(r["variantes"]) - 2
    assert r["decisiones"]["elegida"] == previo["decisiones"]["elegida"]

    # Una segunda ejecución con --reanudar no entrena nada y decide lo mismo.
    contador = []
    m = cargar()
    preparar(m, tmp_path, monkeypatch, contador, ablacion=ablacion, reanudar=True)
    m.main()
    despues = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert contador == []
    assert despues["decisiones"]["elegida"] == r["decisiones"]["elegida"]
    for nombre in r["variantes"]:
        assert despues["variantes"][nombre]["validacion"]["todas"]["logloss"] == pytest.approx(
            r["variantes"][nombre]["validacion"]["todas"]["logloss"], abs=1e-6)


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
