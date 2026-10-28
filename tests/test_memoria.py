import sys

import pytest

from valenbisi.memoria import comprobar_memoria, memoria_libre_gb, memoria_proceso_gb

solo_windows = pytest.mark.skipif(sys.platform != "win32", reason="usa la API de Windows")


@solo_windows
def test_mide_el_proceso_y_el_equipo():
    actual, pico = memoria_proceso_gb()
    assert 0 < actual <= pico
    assert memoria_libre_gb() > 0


@solo_windows
def test_para_si_no_queda_memoria():
    comprobar_memoria(0.0)
    with pytest.raises(MemoryError, match="libres"):
        comprobar_memoria(10_000.0)
