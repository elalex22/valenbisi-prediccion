"""La evaluación final de la v2, de principio a fin, solo con datos inventados.

La evaluación real se ejecuta una vez: los errores de código tienen que salir
aquí. Los zips imitan al clon, incluidos un zip anterior al periodo que ni
siquiera es un zip válido (si se abriera, daría error) y la hora repetida de octubre.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from fabrica import csv_nuevo, escribir_zip, nombre_foto, parquet_features_v2
from valenbisi.evaluacion import FIN_PRUEBA_V2, FRONTERAS_V2, INICIO_SECUNDARIO_V2, limites
from valenbisi.features_v2 import FEATURES_RETARDOS, FEATURES_V2_FINALES
from valenbisi.modelo import FEATURES_MODELO

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluacion_final_v2.py"
ARBOLES = {"lightgbm_v2": 5, "lightgbm_v1": 3}
ESTACION_NUEVA = 99
FECHAS = [*pd.date_range("2026-06-29", "2026-07-14"), *pd.date_range("2026-10-24", "2026-10-29")]


def cargar_script():
    spec = importlib.util.spec_from_file_location("evaluacion_final_v2", SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def foto(momento: pd.Timestamp, rng: np.random.Generator, estaciones: list[int]) -> tuple[str, str]:
    filas = []
    for estacion in estaciones:
        libres = int(rng.integers(0, 8))
        filas.append((estacion, "T", 10 - min(libres, 10), libres, 10, momento.strftime("%d/%m/%Y %H:%M:%S")))
    return nombre_foto(momento.strftime("%d-%m-%Y"), momento.strftime("%H-%M-05")), csv_nuevo(filas)


def clon_inventado(carpeta: Path) -> None:
    rng = np.random.default_rng(1)
    (carpeta / "20-09-2025.zip").write_bytes(b"esto no es un zip: fuera del periodo, nunca se abre")
    estaciones = [1, 2, 3, 4, ESTACION_NUEVA]
    for fecha in FECHAS:
        momentos = pd.date_range(fecha, periods=96, freq="15min")
        escribir_zip(carpeta / f"{fecha:%d-%m-%Y}.zip", [foto(m, rng, estaciones) for m in momentos])


def desarrollo_inventado(tmp_path: Path) -> tuple[Path, Path, Path]:
    features = tmp_path / "features_v2.parquet"
    parquet_features_v2(features, frecuencia="1h")     # todas las horas: la referencia de frecuencia las necesita
    estaciones = tmp_path / "estaciones.parquet"
    pd.DataFrame({"estacion": np.array([1, 2, 3, 4], dtype="int16"), "direccion": "Calle", "lat": [39.46, 39.47, 39.48, 39.49],
                  "lon": [-0.38, -0.37, -0.36, -0.35], "total": np.int16(10)}).to_parquet(estaciones, index=False)
    # La última foto de desarrollo, como en el dataset limpio.
    limpio = tmp_path / "limpio.parquet"
    pd.DataFrame({"ts_utc": [pd.Timestamp("2025-09-24 21:45", tz="UTC")]}).to_parquet(limpio, index=False)
    return features, estaciones, limpio


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    clon = tmp_path / "clon"
    clon.mkdir()
    clon_inventado(clon)
    features, estaciones, limpio = desarrollo_inventado(tmp_path)
    modulo = cargar_script()
    filas = {c: pq.read_table(features, columns=["slot_utc"], filters=limites(c, FRONTERAS_V2)).num_rows
             for c in ("entrenamiento", "validacion")}
    monkeypatch.setattr(modulo, "FILAS_ESPERADAS", filas)
    monkeypatch.setattr(modulo, "ARBOLES", ARBOLES)
    monkeypatch.setattr(modulo, "REPLICAS_BOOTSTRAP", 3)
    monkeypatch.setattr(modulo, "MIN_POSITIVOS_ESTACIONES_NUEVAS", 1)
    monkeypatch.setattr(modulo, "PARAMETROS", {**modulo.PARAMETROS, "min_data_in_leaf": 20, "num_threads": 2})
    monkeypatch.setattr(modulo, "estado_git", lambda: {"commit": "abc1234", "arbol_limpio": True,
                                                       "desde_congelado": True})
    argumentos = ["evaluacion_final_v2.py", "--clon", str(clon), "--features", str(features),
                  "--estaciones", str(estaciones), "--limpio", str(limpio), "--carpeta", str(tmp_path / "prueba"),
                  "--salida", str(tmp_path / "resumen.json"), "--modelos", str(tmp_path / "modelos"),
                  "--figuras", str(tmp_path / "figuras"), "--minimo-libre-gb", "0"]
    return modulo, tmp_path, argumentos


def ejecutar(modulo, monkeypatch, argumentos):
    monkeypatch.setattr(sys, "argv", argumentos)
    modulo.main()


def test_sin_confirmar_no_lee_nada(entorno, monkeypatch):
    modulo, tmp_path, argumentos = entorno
    with pytest.raises(SystemExit):
        ejecutar(modulo, monkeypatch, argumentos)
    assert not (tmp_path / "prueba").exists() and not (tmp_path / "modelos").exists()


@pytest.mark.parametrize("git", [{"arbol_limpio": False, "desde_congelado": True},
                                 {"arbol_limpio": True, "desde_congelado": False}])
def test_se_niega_con_cambios_o_fuera_de_la_congelacion(entorno, monkeypatch, git):
    modulo, tmp_path, argumentos = entorno
    monkeypatch.setattr(modulo, "estado_git", lambda: {"commit": "abc1234", **git})
    with pytest.raises(SystemExit, match="árbol limpio"):
        ejecutar(modulo, monkeypatch, [*argumentos, "--confirmar-prueba"])
    assert not (tmp_path / "prueba").exists()


def test_los_modelos_se_guardan_antes_de_abrir_el_clon(entorno, monkeypatch):
    modulo, tmp_path, argumentos = entorno

    def abrir_clon(*args, **kwargs):
        assert (tmp_path / "modelos" / "lightgbm_v2.txt").exists()
        assert (tmp_path / "modelos" / "lightgbm_v1.txt").exists()
        raise RuntimeError("clon abierto")

    monkeypatch.setattr(modulo, "preparar_filas", abrir_clon)
    with pytest.raises(RuntimeError, match="clon abierto"):
        ejecutar(modulo, monkeypatch, [*argumentos, "--confirmar-prueba"])


def test_si_falla_el_entrenamiento_no_se_abre_el_clon(entorno, monkeypatch):
    modulo, tmp_path, argumentos = entorno

    def falla(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(modulo, "entrenar_fijo", falla)
    with pytest.raises(MemoryError):
        ejecutar(modulo, monkeypatch, [*argumentos, "--confirmar-prueba"])
    assert not (tmp_path / "prueba").exists()


def test_evaluacion_completa_y_segunda_ejecucion(entorno, monkeypatch, capsys):
    modulo, tmp_path, argumentos = entorno
    ejecutar(modulo, monkeypatch, [*argumentos, "--confirmar-prueba"])
    resumen = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    filas = pd.read_parquet(tmp_path / "prueba" / "features.parquet")

    # Periodo: los primeros datos, nada del 29-10 y la etiqueta dentro de él.
    assert filas["slot_utc"].min() == pd.Timestamp("2026-06-29", tz="Europe/Madrid")
    assert (filas["slot_utc"] + pd.Timedelta(minutes=60) < FIN_PRUEBA_V2).all()
    assert filas["slot_utc"].max() == pd.Timestamp("2026-10-28 22:45", tz="Europe/Madrid")
    # La hora repetida de octubre se descarta.
    local = filas["slot_utc"].dt.tz_convert("Europe/Madrid")
    assert not ((local.dt.date == pd.Timestamp("2026-10-25").date()) & (local.dt.hour == 2)).any()
    assert resumen["datos"]["fotos_hora_repetida"] == 4
    assert set(filas.columns) == {"slot_utc", *FEATURES_V2_FINALES, "llena_60"}
    # Los festivos de 2026 entran: el 09-10 no está en estos datos, pero tampoco da error la tabla.
    assert not filas["festivo"].any()

    # La primera semana no tiene retardos de 7 días; la segunda, sí.
    primera = filas["slot_utc"] < INICIO_SECUNDARIO_V2
    assert filas.loc[primera, "libres_obj_7d"].isna().mean() > 0.9
    segunda = filas["slot_utc"].between(pd.Timestamp("2026-07-08", tz="UTC"), pd.Timestamp("2026-07-14", tz="UTC"))
    assert segunda.any() and filas.loc[segunda, FEATURES_RETARDOS].notna().all(axis=None)

    # Los dos resultados, con todos los modelos, las métricas y el logloss.
    principal, secundario = resumen["prueba"]["principal"], resumen["prueba"]["secundario"]
    assert principal["todas"]["lightgbm_v2"]["filas"] == len(filas)
    assert secundario["todas"]["lightgbm_v2"]["filas"] == int((~primera).sum())
    assert {"todas", "llena_ahora", "no_llena_ahora", "0", "1", "2-3", "4-6", "7+", "con_retardos",
            "sin_retardos"} <= set(principal)
    for modelo in modulo.MODELOS:
        assert {"auc_pr", "brier", "brier_skill", "logloss"} <= set(principal["todas"][modelo])
    # Las dos versiones de LightGBM son modelos distintos.
    assert principal["todas"]["lightgbm_v2"]["logloss"] != principal["todas"]["lightgbm_v1"]["logloss"]

    # Intervalos y diferencias frente a las referencias y a la v1, en los dos resultados.
    for escenario in ("principal", "secundario"):
        intervalos = resumen["intervalos_95"][escenario]
        for otro in ("persistencia", "frecuencia", "libres", "lightgbm_v1"):
            assert set(intervalos[f"lightgbm_v2 - {otro}"]) == {"auc_pr", "brier_skill", "logloss"}
        assert isinstance(resumen["frente_a_v1"][escenario]["mejora_clara"], bool)

    # Calibración, mes a mes, y estaciones nuevas aparte.
    assert set(resumen["calibracion"]) == {"libres", "lightgbm_v1", "lightgbm_v2"}
    assert [m["mes"] for m in resumen["calibracion_por_mes"]] == ["2026-06", "2026-07", "2026-10"]
    assert resumen["estaciones_nuevas"]["estaciones"] == [ESTACION_NUEVA]
    assert resumen["estaciones_nuevas"]["filas"] == int((filas["estacion"] == ESTACION_NUEVA).sum())
    assert "lightgbm_v2" in resumen["estaciones_nuevas"]["metricas"]
    assert {p.name for p in (tmp_path / "figuras").iterdir()} == {
        "calibracion_prueba_v2.png", "curvas_pr_prueba_v2.png", "mensual_prueba_v2.png"}
    assert resumen["decisiones"]["features"]["lightgbm_v1"] == FEATURES_MODELO

    # Una segunda ejecución se niega; repetida por error, reutiliza los modelos y da lo mismo.
    with pytest.raises(SystemExit, match="ya se evaluó"):
        ejecutar(modulo, monkeypatch, [*argumentos, "--confirmar-prueba"])

    def no_entrena(*args, **kwargs):
        raise AssertionError("no debería reentrenar")

    monkeypatch.setattr(modulo, "entrenar_fijo", no_entrena)
    ejecutar(modulo, monkeypatch, [*argumentos, "--confirmar-prueba", "--repetir-por-error"])
    repetida = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert repetida["repetida_por_error"] is True
    assert repetida["prueba"] == resumen["prueba"]
    assert repetida["intervalos_95"] == resumen["intervalos_95"]


def test_para_si_el_clon_empieza_cerca_de_desarrollo(tmp_path):
    from valenbisi.prueba_v2 import preparar_filas

    clon = tmp_path / "clon"
    clon.mkdir()
    clon_inventado(clon)
    _, estaciones, _ = desarrollo_inventado(tmp_path)
    with pytest.raises(ValueError, match="pasado"):
        preparar_filas(clon, tmp_path / "prueba", pd.read_parquet(estaciones), FRONTERAS_V2["prueba"], FIN_PRUEBA_V2,
                       ultimo_pasado=pd.Timestamp("2026-06-25", tz="UTC"), minimo_libre_gb=0)
