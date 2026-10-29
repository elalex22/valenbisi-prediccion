"""Meteorología horaria de Open-Meteo (Historical Weather API) y sus features sin fuga.

Datos: reanálisis («best match»: ERA5, ERA5-Land e IFS) de un punto en el
centro de València, en UTC. Licencia CC BY 4.0, con la atribución «Weather data
by Open-Meteo.com». La API gratuita es solo para uso no comercial.

Cómo se fecha cada valor (documentación de Open-Meteo):
- temperatura, humedad, viento y nubosidad son **instantáneos** en la hora indicada;
- la precipitación es la **suma de la hora anterior** a la indicada.

Así, en el instante t se usa la hora ``h = t`` truncada a la hora: todos sus
valores son de h o anteriores, nunca posteriores a t.

Supuesto que se documenta: el reanálisis se calcula días después. En
producción habría que usar observaciones en tiempo real, así que estas features
suponen un «tiempo pasado conocido» algo mejor que el disponible de verdad.
"""

import json
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

URL_ARCHIVO = "https://archive-api.open-meteo.com/v1/archive"
# Plaza del Ayuntamiento de València: todas las estaciones caen a pocos km.
LATITUD, LONGITUD = 39.4699, -0.3763
# Nombre en la API -> nombre de la feature.
VARIABLES = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humedad",
    "wind_speed_10m": "viento",
    "cloud_cover": "nubosidad",
    "precipitation": "prec_1h",
}
HORAS_PREC_3H = 3
# La lluvia del reanálisis es ruidosa en cantidad; «ha llovido algo» puede ser más robusto.
UMBRAL_LLUVIA_MM = 0.1
FEATURES_METEO = ["temp", "humedad", "viento", "nubosidad", "prec_1h", "prec_3h", "llueve_1h"]


def url_archivo(inicio: date, fin: date, latitud: float = LATITUD, longitud: float = LONGITUD) -> str:
    """URL de la Historical Weather API, en UTC y con marcas de tiempo Unix (sin ambigüedad de horario de verano)."""
    parametros = {
        "latitude": latitud, "longitude": longitud,
        "start_date": inicio.isoformat(), "end_date": fin.isoformat(),
        "hourly": ",".join(VARIABLES), "timezone": "GMT", "timeformat": "unixtime",
        "wind_speed_unit": "kmh", "precipitation_unit": "mm",
    }
    return f"{URL_ARCHIVO}?{urlencode(parametros)}"


def leer_respuesta(ruta: Path) -> pd.DataFrame:
    """La respuesta JSON guardada, como una fila por hora: ``hora_utc`` y las variables en float32.

    Comprueba que las horas son únicas, consecutivas y que no falta ninguna variable.
    """
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    horario = datos["hourly"]
    faltan = [v for v in VARIABLES if v not in horario]
    if faltan:
        raise ValueError(f"Faltan variables en la respuesta: {faltan}")
    # Los huecos llegan como null: pasan a NaN, no se rellenan.
    df = pd.DataFrame({"hora_utc": pd.to_datetime(horario["time"], unit="s", utc=True),
                       **{nombre: pd.Series(horario[v], dtype="float64").astype("float32")
                          for v, nombre in VARIABLES.items()}})
    pasos = df["hora_utc"].diff().dropna()
    if not (pasos == pd.Timedelta(hours=1)).all():
        raise ValueError("Las horas de la respuesta no son consecutivas")
    return df


def tiempo_hora_pasada(slot_utc: pd.Series, horario: pd.DataFrame) -> pd.DataFrame:
    """Features meteorológicas de cada instante t, con la hora ``h = t`` truncada a la hora.

    - ``temp``, ``humedad``, ``viento``, ``nubosidad``: valor instantáneo en h.
    - ``prec_1h``: lluvia de la hora anterior a h.
    - ``prec_3h``: lluvia de las 3 horas anteriores a h (h−2, h−1 y h). Si falta
      alguna de las tres, queda NaN: nunca se rellena.
    - ``llueve_1h``: 1 si ``prec_1h`` ≥ ``UMBRAL_LLUVIA_MM``, 0 si no (NaN si falta).
      La ablación decide entre la cantidad y esta versión binaria.
    Si no hay dato para h, todo queda NaN.
    """
    por_hora = horario.set_index("hora_utc")
    prec = por_hora["prec_1h"]
    # Suma de h−2..h solo con las tres horas presentes (los huecos dan NaN).
    completa = prec.reindex(pd.date_range(prec.index.min(), prec.index.max(), freq="h"))
    prec_3h = completa.rolling(HORAS_PREC_3H, min_periods=HORAS_PREC_3H).sum()
    tabla = por_hora[["temp", "humedad", "viento", "nubosidad", "prec_1h"]].join(prec_3h.rename("prec_3h"))
    tabla["llueve_1h"] = (tabla["prec_1h"] >= UMBRAL_LLUVIA_MM).astype("float32").where(tabla["prec_1h"].notna())

    h = pd.DatetimeIndex(slot_utc).floor("h")
    salida = tabla.reindex(h)[FEATURES_METEO].astype("float32")
    salida.index = slot_utc.index
    return salida
