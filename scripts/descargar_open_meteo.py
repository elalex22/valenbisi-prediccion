"""Descarga la meteorología horaria de Open-Meteo para todo el periodo del proyecto.

Una sola llamada a la Historical Weather API (punto: centro de València, UTC).
Guarda la respuesta tal cual en data/raw/open_meteo/, más un fichero con la URL
y la fecha de descarga. No sobrescribe nada: si el fichero existe, se para.

Cubre también el periodo de la prueba v2: es una variable externa, no una
etiqueta. Aun así, de ese periodo solo se informa de cuántas horas hay y cuántos
huecos, nunca de los valores.

Datos: Open-Meteo.com, CC BY 4.0 (uso no comercial de la API gratuita).

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\descargar_open_meteo.py
"""

import argparse
import json
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from valenbisi.open_meteo import leer_respuesta, url_archivo

RAIZ = Path(__file__).resolve().parents[1]
# Un día antes del primer dato, para que prec_3h exista desde el principio.
INICIO, FIN = date(2022, 11, 30), date(2026, 10, 28)
INICIO_PRUEBA_V2 = pd.Timestamp("2025-09-25", tz="Europe/Madrid").tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--carpeta", type=Path, default=RAIZ / "data" / "raw" / "open_meteo")
    args = parser.parse_args()
    salida = args.carpeta / f"valencia_horario_{INICIO}_{FIN}.json"
    if salida.exists():
        sys.exit(f"{salida} ya existe: no se sobrescribe.")

    url = url_archivo(INICIO, FIN)
    print(f"Descargando {url}", flush=True)
    with urllib.request.urlopen(url, timeout=300) as respuesta:
        contenido = respuesta.read()
    args.carpeta.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(contenido)
    salida.with_suffix(".origen.json").write_text(json.dumps(
        {"url": url, "descargado": datetime.now().isoformat(timespec="seconds"),
         "licencia": "CC BY 4.0, Weather data by Open-Meteo.com"}, ensure_ascii=False, indent=2), encoding="utf-8")

    datos = json.loads(contenido)
    horario = leer_respuesta(salida)
    print(f"Guardado en {salida} ({len(contenido) / 1e6:.1f} MB)")
    print(f"Punto devuelto: {datos.get('latitude')}, {datos.get('longitude')}, elevación {datos.get('elevation')} m")
    print(f"Horas: {len(horario)} ({horario['hora_utc'].min()} a {horario['hora_utc'].max()})")

    desarrollo = horario[horario["hora_utc"] < INICIO_PRUEBA_V2]
    prueba_v2 = horario[horario["hora_utc"] >= INICIO_PRUEBA_V2]
    variables = [c for c in horario.columns if c != "hora_utc"]
    print(f"\nDesarrollo ({len(desarrollo)} horas): huecos y valores")
    print(desarrollo[variables].describe(percentiles=[0.5]).T.assign(huecos=desarrollo[variables].isna().sum())
          .round(2).to_string())
    lluvia = desarrollo.groupby(desarrollo["hora_utc"].dt.year)["prec_1h"].sum().round(0)
    print(f"Lluvia total por año (mm): {lluvia.to_dict()}")
    print(f"\nPrueba v2 ({len(prueba_v2)} horas): solo huecos, sin mirar valores")
    print(prueba_v2[variables].isna().sum().to_dict())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
