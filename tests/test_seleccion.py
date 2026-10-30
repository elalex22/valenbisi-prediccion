"""Reglas de elección de configuraciones y modelos de la v2."""

from valenbisi.seleccion import elegir_configuracion, es_clara


def dif(estimacion: float, inferior: float, superior: float) -> dict:
    return {"estimacion": estimacion, "inferior": inferior, "superior": superior}


def test_una_mejora_es_clara_si_supera_el_doble_del_ruido_y_el_intervalo_excluye_el_0():
    assert es_clara(dif(-0.0005, -0.0008, -0.0002), ruido=0.0001)
    assert not es_clara(dif(-0.0005, -0.0008, 0.0001), ruido=0.0001)      # el intervalo cruza el 0
    assert not es_clara(dif(-0.00015, -0.0002, -0.0001), ruido=0.0001)    # menos del doble del ruido
    assert not es_clara(dif(0.0003, 0.0001, 0.0005), ruido=0.0)           # empeora
    assert not es_clara(None, ruido=0.0)


def test_la_configuracion_de_partida_se_queda_salvo_mejora_clara():
    assert elegir_configuracion("L3", "L0", dif(-0.0005, -0.0008, -0.0002), ruido=0.000064) == "L3"
    assert elegir_configuracion("L3", "L0", dif(-0.0001, -0.0003, -0.00001), ruido=0.000064) == "L0"   # < 2 × ruido
    assert elegir_configuracion("L3", "L0", dif(-0.0005, -0.0009, 0.00002), ruido=0.000064) == "L0"   # cruza el 0
    assert elegir_configuracion("L0", "L0", None, ruido=0.000064) == "L0"
