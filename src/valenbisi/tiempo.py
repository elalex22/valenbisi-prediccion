"""Lectura de los datos diarios de AEMET (estación 8416, València).

Cada JSON trae una lista de días, en latin-1, con los números como texto y coma
decimal. ``prec="Ip"`` es precipitación inapreciable (menos de 0,1 mm). Algunos
ficheros solo contienen ``{"estado": 404, "descripcion": "datos expirados"}``:
se saltan.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

VARIABLES = ["tmed", "tmin", "tmax", "prec", "sol"]
# Inapreciable: ha llovido algo, pero menos que el mínimo medible (0,1 mm).
PREC_INAPRECIABLE = 0.05


def a_numero(valores: pd.Series, variable: str) -> pd.Series:
    """Texto con coma decimal a float32; da error ante cualquier texto que no se conozca."""
    valores = valores.astype("string")  # una columna que falta entera llega sin tipo texto
    texto = valores.str.replace(",", ".", regex=False)
    if variable == "prec":
        texto = texto.replace("Ip", str(PREC_INAPRECIABLE))
    numero = pd.to_numeric(texto, errors="coerce")
    raros = numero.isna() & valores.notna()
    if raros.any():
        raise ValueError(f"Valor no numérico en {variable}: {valores[raros].iloc[0]!r}")
    return numero.astype("float32")


def leer_json(ruta: Path) -> pd.DataFrame:
    """Los días de un fichero, con ``fecha`` y las variables numéricas (vacío si es un 404)."""
    datos = json.loads(Path(ruta).read_text(encoding="latin-1"))
    if isinstance(datos, dict):
        if datos.get("estado") == 404:
            return pd.DataFrame(columns=["fecha", *VARIABLES])
        raise ValueError(f"Respuesta inesperada en {ruta}: {str(datos)[:80]}")
    df = pd.DataFrame(datos).reindex(columns=["fecha", *VARIABLES])
    return pd.DataFrame({
        "fecha": pd.to_datetime(df["fecha"], format="%Y-%m-%d"),
        **{v: a_numero(df[v], v) for v in VARIABLES},
    })


def leer_tiempo(carpeta: Path) -> pd.DataFrame:
    """Una fila por día con las variables de AEMET, ordenada por ``fecha``.

    Da error si un mismo día aparece en dos ficheros.
    """
    trozos = [leer_json(ruta) for ruta in sorted(Path(carpeta).glob("*.json"))]
    df = pd.concat([t for t in trozos if not t.empty], ignore_index=True)
    if df["fecha"].duplicated().any():
        raise ValueError(f"Día repetido en los datos de AEMET: {df.loc[df['fecha'].duplicated(), 'fecha'].iloc[0]}")
    return df.sort_values("fecha", ignore_index=True).astype({v: np.float32 for v in VARIABLES})
