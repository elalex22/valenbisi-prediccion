import json

import pandas as pd
import pytest

from valenbisi.tiempo import PREC_INAPRECIABLE, leer_json, leer_tiempo


def dia(fecha: str, prec: str = "0,0", **otros) -> dict:
    return {"fecha": fecha, "indicativo": "8416", "nombre": "VALÈNCIA", "tmed": "14,4", "prec": prec,
            "tmin": "9,8", "tmax": "19,0", "sol": "5,6", **otros}


def escribir(ruta, datos) -> None:
    ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="latin-1")


def test_coma_decimal_e_inapreciable(tmp_path):
    escribir(tmp_path / "t.json", [dia("2024-01-01"), dia("2024-01-02", prec="Ip"), dia("2024-01-03", prec="12,3")])
    df = leer_json(tmp_path / "t.json")

    assert df["fecha"].tolist() == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    assert df["tmed"].iloc[0] == pytest.approx(14.4)
    assert df["prec"].tolist() == pytest.approx([0.0, PREC_INAPRECIABLE, 12.3])
    assert df["prec"].dtype == "float32"


def test_variable_ausente_queda_nan(tmp_path):
    registro = dia("2024-01-01")
    del registro["sol"]
    escribir(tmp_path / "t.json", [registro])
    assert leer_json(tmp_path / "t.json")["sol"].isna().all()


def test_texto_desconocido_da_error(tmp_path):
    escribir(tmp_path / "t.json", [dia("2024-01-01", prec="Acum")])
    with pytest.raises(ValueError, match="prec"):
        leer_json(tmp_path / "t.json")


def test_fichero_404_se_salta(tmp_path):
    escribir(tmp_path / "a.json", {"descripcion": "datos expirados", "estado": 404})
    escribir(tmp_path / "b.json", [dia("2024-07-02"), dia("2024-07-01")])
    df = leer_tiempo(tmp_path)

    assert df["fecha"].tolist() == list(pd.to_datetime(["2024-07-01", "2024-07-02"]))


def test_dia_repetido_da_error(tmp_path):
    escribir(tmp_path / "a.json", [dia("2024-07-01")])
    escribir(tmp_path / "b.json", [dia("2024-07-01")])
    with pytest.raises(ValueError, match="repetido"):
        leer_tiempo(tmp_path)
