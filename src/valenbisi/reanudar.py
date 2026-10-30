"""Reutilizar entrenamientos de una ejecución anterior (opción ``--reanudar`` de los scripts largos).

Un entrenamiento se identifica por sus columnas (en orden), la semilla, los
parámetros completos y la parada temprana. Solo se reutiliza si coincide todo.
"""

import json


def clave(features: list[str], semilla: int, parametros: dict, max_arboles: int, paciencia: int) -> str:
    """Identifica un entrenamiento: columnas (en orden), semilla, parámetros y parada temprana."""
    return json.dumps([list(features), semilla, parametros, max_arboles, paciencia], sort_keys=True)


def indice_previas(resumen: dict, parada_por_defecto: dict | None = None) -> dict[str, tuple[str, bool]]:
    """{clave: (variante, parada_asumida)} de las variantes entrenadas de un resumen anterior.

    Las variantes sin ``features`` (simulaciones) no se indexan: se reutilizan
    junto con su modelo. Si una variante no registra la parada temprana, se toma
    de ``parada_por_defecto`` y se marca como asumida; sin él, no se indexa.
    """
    indice = {}
    for nombre, v in resumen.get("variantes", {}).items():
        if "features" not in v:
            continue
        asumida = "max_arboles" not in v or "paciencia" not in v
        if asumida and parada_por_defecto is None:
            continue
        parada = parada_por_defecto or {}
        k = clave(v["features"], v["semilla"], v.get("parametros", resumen.get("parametros")),
                  v.get("max_arboles", parada.get("max_arboles")), v.get("paciencia", parada.get("paciencia")))
        indice.setdefault(k, (nombre, asumida))
    return indice


def comprobar_nombre_nuevo(nombre: str, registradas) -> None:
    """Error si ``nombre`` ya está registrado: se sobrescribirían sus predicciones."""
    if nombre in registradas:
        raise ValueError(f"El nombre {nombre!r} ya está registrado: se sobrescribirían sus predicciones")
