"""Reutilización de entrenamientos (--reanudar) y la lista de features finales de la v2."""

import json
from pathlib import Path

import pytest

from valenbisi.features_v2 import FEATURES_V2, FEATURES_V2_FINALES
from valenbisi.modelo import FEATURES_MODELO
from valenbisi.reanudar import clave, comprobar_nombre_nuevo, indice_previas

RESUMEN_ABLACION = Path(__file__).resolve().parents[1] / "data" / "processed" / "resumen_ablacion_v2.json"


def test_clave_distingue_orden_semilla_parametros_y_parada():
    base = clave(["a", "b"], 42, {"lr": 0.05}, 3000, 100)
    assert base == clave(["a", "b"], 42, {"lr": 0.05}, 3000, 100)
    otras = [clave(["b", "a"], 42, {"lr": 0.05}, 3000, 100), clave(["a", "b"], 43, {"lr": 0.05}, 3000, 100),
             clave(["a", "b"], 42, {"lr": 0.1}, 3000, 100), clave(["a", "b"], 42, {"lr": 0.05}, 2000, 100),
             clave(["a", "b"], 42, {"lr": 0.05}, 3000, 50)]
    assert base not in otras


def test_sin_parada_registrada_solo_se_indexa_con_un_valor_por_defecto():
    previo = {"parametros": {"lr": 0.05}, "variantes": {
        "A0": {"features": ["a"], "semilla": 42},
        "B": {"features": ["b"], "semilla": 42, "parametros": {"lr": 0.1}, "max_arboles": 10, "paciencia": 5},
        "B_sim": {"simulacion": "x", "de": "B"},
    }}
    assert indice_previas(previo) == {clave(["b"], 42, {"lr": 0.1}, 10, 5): ("B", False)}
    con_defecto = indice_previas(previo, {"max_arboles": 3000, "paciencia": 100})
    assert con_defecto[clave(["a"], 42, {"lr": 0.05}, 3000, 100)] == ("A0", True)
    assert len(con_defecto) == 2


def test_un_nombre_repetido_es_un_error():
    comprobar_nombre_nuevo("L1", {"L0": {}})
    with pytest.raises(ValueError, match="ya está registrado"):
        comprobar_nombre_nuevo("L0", {"L0": {}})


def test_features_finales_de_la_v2():
    assert len(FEATURES_V2_FINALES) == 23
    assert FEATURES_V2_FINALES == [c for c in FEATURES_V2 if c in set(FEATURES_V2_FINALES)]   # orden de FEATURES_V2
    assert set(FEATURES_MODELO) <= set(FEATURES_V2_FINALES)


@pytest.mark.skipif(not RESUMEN_ABLACION.exists(), reason="sin el resumen de la ablación v2 (datos no versionados)")
def test_features_finales_coinciden_con_la_ablacion():
    resumen = json.loads(RESUMEN_ABLACION.read_text(encoding="utf-8"))
    assert resumen["decisiones"]["features_finales"] == FEATURES_V2_FINALES
