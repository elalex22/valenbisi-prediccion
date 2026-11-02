"""Experimento: los días como códigos enteros dan el mismo bootstrap que como texto."""

import numpy as np
import pandas as pd

from valenbisi.evaluacion import ZONA, intervalos_bootstrap
from valenbisi.experimento import codigos_de_dia


def test_un_codigo_por_dia_local_tambien_en_los_cambios_de_hora():
    # 30-03-2025 y 26-10-2025: días de 23 y 25 h en Madrid.
    slots = pd.Series(pd.date_range("2025-03-29 20:00", "2025-03-31 02:00", freq="15min", tz="UTC").append(
        pd.date_range("2025-10-25 20:00", "2025-10-27 02:00", freq="15min", tz="UTC")))
    texto = slots.dt.tz_convert(ZONA).dt.strftime("%Y-%m-%d").to_numpy()
    codigos = codigos_de_dia(slots)
    assert codigos.dtype == np.int32
    np.testing.assert_array_equal(codigos, pd.factorize(texto)[0])


def test_el_bootstrap_con_codigos_es_identico_al_de_texto():
    rng = np.random.default_rng(0)
    slots = pd.Series(pd.to_datetime(np.sort(rng.integers(1_740_000_000, 1_760_000_000, 5000)), unit="s", utc=True))
    texto = slots.dt.tz_convert(ZONA).dt.strftime("%Y-%m-%d").to_numpy()
    codigos = codigos_de_dia(slots)
    y = rng.random(5000) < 0.1
    pares = {"a": rng.random(5000) * 0.3, "b": rng.random(5000) * 0.3}
    filtro = rng.random(5000) < 0.5                      # como en los subgrupos de Experimento.comparar
    for f in (np.ones(5000, dtype=bool), filtro):
        args = dict(diferencias=[("a", "b")], n=20, medidas=("logloss",), decimales=6)
        con_texto = intervalos_bootstrap(y[f], {k: v[f] for k, v in pares.items()}, texto[f], 0.1, **args)
        con_codigos = intervalos_bootstrap(y[f], {k: v[f] for k, v in pares.items()}, codigos[f], 0.1, **args)
        assert con_texto == con_codigos
