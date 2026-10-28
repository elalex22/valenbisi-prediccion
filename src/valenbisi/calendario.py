"""Features de calendario del instante t: hora del día, día de la semana, mes y festivos.

Todo sale de ``slot_utc`` pasado a hora local, no de ``ts``: ``slot_utc`` ya está
redondeado al cuarto de hora. El calendario se conoce de antemano, así que no
hay riesgo de fuga.
"""

import pandas as pd

from valenbisi.evaluacion import ZONA

# Festivos de la ciudad de València en el periodo de los datos: nacionales,
# autonómicos (Comunitat Valenciana) y locales. Los que caen en domingo y no se
# trasladan se dejan por completitud. Fuentes:
# - 2022: decreto autonómico; el lunes 26-12 no fue festivo (se cambió por el 24-06).
# - 2023: los dos locales son San Vicente Ferrer (17-04) y el traslado de San
#   Vicente Mártir (domingo 22-01) al 24-04, aprobado en pleno municipal.
#   El 24-04-2023 no está confirmado en fuente oficial (solo prensa y
#   calendarios de terceros); se cuenta como festivo igualmente.
# - 2024: Decreto 164/2023 del Consell; locales 22-01 y 08-04.
# - 2025: Decreto 100/2024 del Consell; locales 22-01 y 28-04.
# El 24 de junio (San Juan) es festivo «recuperable» en 2024 y 2025: no se trabaja, cuenta.
FESTIVOS = pd.to_datetime([
    "2022-12-06", "2022-12-08",
    "2023-01-06", "2023-04-07", "2023-04-10", "2023-04-17", "2023-04-24", "2023-05-01", "2023-06-24",
    "2023-08-15", "2023-10-09", "2023-10-12", "2023-11-01", "2023-12-06", "2023-12-08", "2023-12-25",
    "2024-01-01", "2024-01-06", "2024-01-22", "2024-03-19", "2024-03-29", "2024-04-01", "2024-04-08",
    "2024-05-01", "2024-06-24", "2024-08-15", "2024-10-09", "2024-10-12", "2024-11-01", "2024-12-06",
    "2024-12-25",
    "2025-01-01", "2025-01-06", "2025-01-22", "2025-03-19", "2025-04-18", "2025-04-21", "2025-04-28",
    "2025-05-01", "2025-06-24", "2025-08-15", "2025-10-09", "2025-11-01", "2025-12-06", "2025-12-08",
    "2025-12-25",
])
# Fuera de este rango la tabla no sabe qué días son festivos.
COBERTURA = (pd.Timestamp("2022-12-01"), pd.Timestamp("2025-12-31"))


def features_calendario(slot_utc: pd.Series) -> pd.DataFrame:
    """``minuto_del_dia`` (0–1425), ``dia_semana`` (0 = lunes), ``mes`` y ``festivo`` en hora local."""
    local = slot_utc.dt.tz_convert(ZONA)
    dia = local.dt.tz_localize(None).dt.normalize()
    if dia.min() < COBERTURA[0] or dia.max() > COBERTURA[1]:
        raise ValueError(f"La tabla de festivos solo cubre {COBERTURA[0].date()} – {COBERTURA[1].date()}")
    return pd.DataFrame({
        "minuto_del_dia": (local.dt.hour * 60 + local.dt.minute).astype("int16"),
        "dia_semana": local.dt.day_of_week.astype("int8"),
        "mes": local.dt.month.astype("int8"),
        "festivo": dia.isin(FESTIVOS),
    }, index=slot_utc.index)
