"""Reglas para elegir configuraciones y modelos en la validación v2 (fijadas en CLAUDE.md antes de ejecutar).

Las diferencias son de logloss (nuevo − actual): negativa = el nuevo mejora.
Cada una es un dict con ``estimacion``, ``inferior`` y ``superior`` (intervalo
al 95 % por bootstrap de días).
"""


def es_clara(diferencia: dict | None, ruido: float) -> bool:
    """Mejora más del doble del ruido de semilla y con el intervalo por debajo del 0."""
    if diferencia is None:
        return False
    e = diferencia["estimacion"]
    return e < 0 and -e > 2 * ruido and diferencia["superior"] < 0


def elegir_configuracion(mejor: str, partida: str, mejor_menos_partida: dict | None, ruido: float) -> str:
    """La configuración de menor logloss sustituye a la de partida solo si la mejora es clara."""
    if mejor == partida:
        return partida
    return mejor if es_clara(mejor_menos_partida, ruido) else partida
