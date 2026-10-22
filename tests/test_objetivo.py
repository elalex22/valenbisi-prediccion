import json

import pandas as pd
import pytest

from valenbisi.dataset import COLUMNAS_SALIDA
from valenbisi.objetivo import (
    alinear_a_slot,
    comprobar_sin_colisiones,
    construir_objetivo,
    etiquetar,
    formatear_resumen_objetivo,
)


def limpio(*registros, desfase_s: int = 2) -> pd.DataFrame:
    """Filas con las columnas del dataset limpio.

    registros: (hora UTC 'YYYY-MM-DD HH:MM', estacion, libres). Las fotos llegan
    ``desfase_s`` segundos tarde, como en los datos reales.
    """
    df = pd.DataFrame(registros, columns=["hora", "estacion", "libres"])
    ts_utc = (pd.to_datetime(df["hora"]) + pd.Timedelta(seconds=desfase_s)).dt.tz_localize("UTC")
    out = pd.DataFrame({
        "ts_utc": ts_utc.astype("datetime64[us, UTC]"),
        "ts": ts_utc.dt.tz_convert("Europe/Madrid").dt.tz_localize(None).astype("datetime64[us]"),
        "estacion": df["estacion"].astype("int16"),
        "bicis": (15 - df["libres"]).astype("int16"),
        "libres": df["libres"].astype("int16"),
        "total": pd.Series(15, index=df.index, dtype="int16"),
        "fecha_actualizacion": (ts_utc.dt.tz_localize(None) - pd.Timedelta(minutes=5)).astype("datetime64[us]"),
    })
    out["llena"] = out["libres"].eq(0)
    return out.sort_values(["ts_utc", "estacion"], ignore_index=True)[COLUMNAS_SALIDA]


def con_slot(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(slot_utc=alinear_a_slot(df["ts_utc"]))


def etiquetas(df: pd.DataFrame) -> dict:
    """{(hora UTC 'HH:MM', estacion): llena_60} de las filas con etiqueta."""
    etiquetadas, _ = etiquetar(con_slot(df))
    return {
        (s.strftime("%H:%M"), int(e)): bool(y)
        for s, e, y in zip(etiquetadas["slot_utc"], etiquetadas["estacion"], etiquetadas["llena_60"])
    }


# Una estación cada 15 min de 10:00 a 11:15 UTC; llena (0 libres) a las 11:00.
SERIE = [(f"2024-01-15 {h}", 1, libres) for h, libres in
         [("10:00", 3), ("10:15", 2), ("10:30", 1), ("10:45", 1), ("11:00", 0), ("11:15", 2)]]


def test_alinear_redondea_al_intervalo_mas_cercano():
    ts = pd.Series(pd.to_datetime(["2024-01-15 10:00:02", "2024-01-15 10:14:58", "2024-01-15 10:33:18"])).dt.tz_localize("UTC")
    assert alinear_a_slot(ts).dt.strftime("%H:%M").tolist() == ["10:00", "10:15", "10:30"]


def test_alinear_da_error_si_la_foto_esta_lejos_del_intervalo():
    ts = pd.Series(pd.to_datetime(["2024-01-15 10:06:00"])).dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="intervalo"):
        alinear_a_slot(ts)


def test_colision_en_el_mismo_intervalo_da_error():
    df = con_slot(pd.concat([limpio(("2024-01-15 10:00", 1, 3), desfase_s=2),
                             limpio(("2024-01-15 10:00", 1, 3), desfase_s=-3)]))
    with pytest.raises(ValueError, match="mismo intervalo"):
        comprobar_sin_colisiones(df)


def test_etiqueta_es_llena_exactamente_60_min_despues():
    assert etiquetas(limpio(*SERIE)) == {("10:00", 1): True, ("10:15", 1): False}


def test_sin_foto_a_t60_no_hay_etiqueta_y_se_cuenta_el_motivo():
    df = limpio(
        ("2024-01-15 10:00", 1, 3), ("2024-01-15 10:00", 2, 3),
        ("2024-01-15 10:15", 1, 3),                              # su t+60 (11:15) no existe
        ("2024-01-15 11:00", 1, 0),                              # la foto de 11:00 no trae la estación 2
    )
    etiquetadas, conteo = etiquetar(con_slot(df))
    assert list(zip(etiquetadas["slot_utc"].dt.strftime("%H:%M"), etiquetadas["estacion"])) == [("10:00", 1)]
    # Sin etiqueta: 10:00 est. 2 (falta la estación) y 10:15 y 11:00 (falta la foto).
    assert conteo == {"sin_etiqueta_falta_foto": 2, "sin_etiqueta_falta_estacion": 1}


def test_hueco_en_t60_no_se_rellena_con_fotos_vecinas():
    df = limpio(("2024-01-15 10:00", 1, 3),
                ("2024-01-15 10:45", 1, 0),   # t+45, llena
                ("2024-01-15 11:15", 1, 0))   # t+75, llena; t+60 no existe
    assert ("10:00", 1) not in etiquetas(df)


def test_la_etiqueta_es_de_la_misma_estacion():
    df = limpio(("2024-01-15 10:00", 1, 3), ("2024-01-15 10:00", 2, 3),
                ("2024-01-15 11:00", 1, 4), ("2024-01-15 11:00", 2, 0))
    assert etiquetas(df) == {("10:00", 1): False, ("10:00", 2): True}


def test_perturbar_el_futuro_solo_cambia_la_etiqueta_de_t():
    """Prueba de fuga: solo t+60 puede influir en la fila de t, y solo en llena_60."""
    base = limpio(*SERIE, *[(h, 2, 5) for h, _, _ in SERIE])
    referencia, _ = etiquetar(con_slot(base))
    fila_10 = (referencia["slot_utc"].dt.strftime("%H:%M") == "10:00") & (referencia["estacion"] == 1)

    def etiquetar_con(hora: str, estacion: int, libres: int) -> pd.DataFrame:
        df = base.copy()
        cambiar = (df["ts_utc"].dt.strftime("%H:%M") == hora) & (df["estacion"] == estacion)
        df.loc[cambiar, "libres"] = libres
        df.loc[cambiar, "bicis"] = 15 - libres
        df.loc[cambiar, "llena"] = libres == 0
        return etiquetar(con_slot(df))[0]

    # t+15, t+30 y t+45 no influyen en nada.
    for hora in ("10:15", "10:30", "10:45"):
        pd.testing.assert_frame_equal(
            etiquetar_con(hora, 1, 0).drop(columns=["bicis", "libres", "llena", "fecha_actualizacion"])
            .loc[fila_10.to_numpy()],
            referencia.drop(columns=["bicis", "libres", "llena", "fecha_actualizacion"]).loc[fila_10.to_numpy()],
        )
    # t+60 solo cambia llena_60 de la fila de t de esa estación.
    cambiado = etiquetar_con("11:00", 1, 4)
    distintas = (cambiado != referencia)
    assert distintas["llena_60"].tolist() == fila_10.tolist()
    assert not distintas.drop(columns=["llena_60", "bicis", "libres", "llena"]).any().any()
    # Las columnas de t (no la etiqueta) siguen siendo las de t.
    assert not distintas.loc[fila_10, ["bicis", "libres", "llena"]].any().any()
    # Cambiar otra estación no afecta a la estación 1.
    otra = etiquetar_con("11:00", 2, 0)
    pd.testing.assert_series_equal(
        otra.loc[otra["estacion"] == 1, "llena_60"], referencia.loc[referencia["estacion"] == 1, "llena_60"]
    )


def test_t60_se_cuenta_en_utc_en_el_cambio_de_hora_de_marzo():
    # 30-03-2025: 01:30 CET = 00:30 UTC; 60 min reales después son las 03:30 CEST = 01:30 UTC.
    df = limpio(("2025-03-30 00:30", 1, 3), ("2025-03-30 01:30", 1, 0))
    etiquetadas, _ = etiquetar(con_slot(df))
    assert etiquetadas["ts"].dt.strftime("%H:%M").tolist() == ["01:30"]
    assert etiquetadas["llena_60"].tolist() == [True]


def test_columnas_de_salida_solo_anaden_slot_y_etiqueta():
    etiquetadas, _ = etiquetar(con_slot(limpio(*SERIE)))
    assert list(etiquetadas.columns) == [*COLUMNAS_SALIDA, "slot_utc", "llena_60"]


def _parquet_inventado(ruta):
    """Dos estaciones, 3 h de fotos, con un hueco y una foto sin la estación 2."""
    horas = pd.date_range("2024-01-15 10:00", "2024-01-15 13:00", freq="15min").strftime("%Y-%m-%d %H:%M")
    registros = []
    for i, h in enumerate(horas):
        if h.endswith("11:30"):
            continue  # hueco: falta la foto entera
        registros.append((h, 1, i % 4))  # llena cada 4 fotos
        if not h.endswith("12:15"):
            registros.append((h, 2, 0 if i % 3 == 0 else 5))
    limpio(*registros).to_parquet(ruta, index=False, row_group_size=7)
    return len(registros)


def test_por_lotes_da_lo_mismo_que_de_golpe(tmp_path):
    _parquet_inventado(tmp_path / "limpio.parquet")
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    de_golpe = construir_objetivo(tmp_path / "limpio.parquet", tmp_path / "a" / "obj.parquet", filas_por_lote=10**6)
    por_lotes = construir_objetivo(tmp_path / "limpio.parquet", tmp_path / "b" / "obj.parquet", filas_por_lote=1)

    assert de_golpe == por_lotes
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a" / "obj.parquet"),
                                  pd.read_parquet(tmp_path / "b" / "obj.parquet"))


def test_resumen_cuadra(tmp_path):
    n = _parquet_inventado(tmp_path / "limpio.parquet")
    resumen = construir_objetivo(tmp_path / "limpio.parquet", tmp_path / "obj.parquet", filas_por_lote=5)
    datos = pd.read_parquet(tmp_path / "obj.parquet")

    assert resumen["filas_entrada"] == n
    assert resumen["con_etiqueta"] == len(datos)
    assert n == resumen["con_etiqueta"] + resumen["sin_etiqueta_falta_foto"] + resumen["sin_etiqueta_falta_estacion"]
    assert resumen["sin_etiqueta_falta_estacion"] == 1  # 11:15 est. 2 (su t+60 es 12:15)
    assert resumen["porcentaje_positivos"] == round(datos["llena_60"].mean() * 100, 2)
    assert resumen["llena_ahora"]["filas"] == int(datos["llena"].sum())
    assert resumen["llena_ahora"]["porcentaje_llena_60"] == round(datos.loc[datos["llena"], "llena_60"].mean() * 100, 2)
    assert json.loads((tmp_path / "resumen_objetivo.json").read_text(encoding="utf-8")) == resumen
    assert "% llena a t+60" in formatear_resumen_objetivo(resumen)
