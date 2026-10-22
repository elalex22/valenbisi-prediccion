"""Etiqueta objetivo: ¿estará llena la estación exactamente 60 minutos después?

Cada fila (estación, instante t) del dataset limpio recibe ``llena_60``, que es
el ``llena`` de la misma estación en ``slot_utc + 60 min``. Si esa fila no
existe, la fila se descarta: nunca se rellena con la foto de t+45 ni de t+75.

De t+60 solo se copia ``llena_60``, ningún otro dato, para que ninguna feature
pueda usar información del futuro por error.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from valenbisi.dataset import EscritorPorBloques, formatear_entero

SLOT = "15min"
HORIZONTE = pd.Timedelta(minutes=60)
MAX_DESFASE = pd.Timedelta(minutes=5)


def alinear_a_slot(ts_utc: pd.Series) -> pd.Series:
    """Redondea al múltiplo de 15 min más cercano (las fotos llegan con segundos de retraso).

    Da error si alguna foto está a más de 5 min de su intervalo: indicaría un
    problema en los datos que conviene ver, no esconder.
    """
    slot = ts_utc.dt.round(SLOT)
    desfase = (ts_utc - slot).abs()
    if (desfase > MAX_DESFASE).any():
        peor = ts_utc.iloc[desfase.to_numpy().argmax()]
        raise ValueError(f"Foto a más de {MAX_DESFASE} de su intervalo de 15 min: {peor}")
    return slot


def comprobar_sin_colisiones(df: pd.DataFrame) -> None:
    """Dos filas en el mismo (slot_utc, estacion) harían ambigua la etiqueta."""
    repetidas = df.duplicated(["slot_utc", "estacion"])
    if repetidas.any():
        fila = df.loc[repetidas.to_numpy(), ["ts_utc", "estacion"]].iloc[0].tolist()
        raise ValueError(f"Dos fotos caen en el mismo intervalo de 15 min para la estación: {fila}")


def etiquetar(filas: pd.DataFrame, referencia: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Añade ``llena_60`` a ``filas`` buscando en ``referencia`` (por defecto, las propias filas).

    Ambos DataFrames deben tener ya ``slot_utc``. Devuelve solo las filas con
    etiqueta y cuántas se descartaron por cada motivo.
    """
    referencia = filas if referencia is None else referencia
    futuro = pd.DataFrame({
        # La fila de t+60 aporta la etiqueta a la fila de t.
        "slot_utc": referencia["slot_utc"] - HORIZONTE,
        "estacion": referencia["estacion"],
        "llena_60": referencia["llena"],
    })
    con_futuro = filas.merge(futuro, on=["slot_utc", "estacion"], how="left", validate="one_to_one")
    sin_etiqueta = con_futuro["llena_60"].isna()
    falta_foto = sin_etiqueta & ~(con_futuro["slot_utc"] + HORIZONTE).isin(referencia["slot_utc"].unique())
    conteo = {
        "sin_etiqueta_falta_foto": int(falta_foto.sum()),
        "sin_etiqueta_falta_estacion": int((sin_etiqueta & ~falta_foto).sum()),
    }
    etiquetadas = con_futuro[~sin_etiqueta].astype({"llena_60": "bool"})
    return etiquetadas.reset_index(drop=True), conteo


def construir_objetivo(
    ruta_limpio: Path,
    ruta_salida: Path,
    filas_por_lote: int = 500_000,
    progreso: Callable[[int], None] | None = None,
) -> dict:
    """Etiqueta el Parquet limpio por lotes y escribe ``ruta_salida`` con su resumen al lado.

    El Parquet limpio está ordenado por ``ts_utc``. Cada lote se junta con las
    filas pendientes del anterior. Se etiquetan las filas cuyo t+60 es anterior
    a la última foto del lote, y por tanto está entero en memoria. El resto
    (la última hora) espera al siguiente lote.
    """
    ruta_salida = Path(ruta_salida)
    escritor = EscritorPorBloques(ruta_salida, filas_por_lote)
    conteo = {"filas_entrada": 0, "con_etiqueta": 0, "sin_etiqueta_falta_foto": 0, "sin_etiqueta_falta_estacion": 0}
    cruce = {(ahora, despues): 0 for ahora in (True, False) for despues in (True, False)}
    columnas = None

    def procesar(filas: pd.DataFrame, referencia: pd.DataFrame) -> None:
        etiquetadas, descartes = etiquetar(filas, referencia)
        for clave, valor in descartes.items():
            conteo[clave] += valor
        if etiquetadas.empty:
            return
        conteo["con_etiqueta"] += len(etiquetadas)
        for (ahora, despues), n in etiquetadas.groupby(["llena", "llena_60"]).size().items():
            cruce[(bool(ahora), bool(despues))] += int(n)
        escritor.agregar(etiquetadas[columnas])

    pendiente = None
    try:
        for i, lote in enumerate(pq.ParquetFile(Path(ruta_limpio)).iter_batches(batch_size=filas_por_lote), start=1):
            lote = lote.to_pandas()
            if columnas is None:
                columnas = [*lote.columns, "slot_utc", "llena_60"]
            conteo["filas_entrada"] += len(lote)
            lote["slot_utc"] = alinear_a_slot(lote["ts_utc"])
            buffer = lote if pendiente is None else pd.concat([pendiente, lote], ignore_index=True)
            comprobar_sin_colisiones(buffer)
            if not buffer["slot_utc"].is_monotonic_increasing:
                raise ValueError("El Parquet limpio no está ordenado por ts_utc")

            ultimo = buffer["slot_utc"].iloc[-1]
            listas = buffer["slot_utc"] + HORIZONTE < ultimo
            procesar(buffer[listas], buffer)
            pendiente = buffer[~listas]
            if progreso:
                progreso(i)

        if pendiente is not None and not pendiente.empty:
            procesar(pendiente, pendiente)
        if conteo["con_etiqueta"] == 0:
            raise ValueError(f"Ninguna fila tiene etiqueta en {ruta_limpio}")
        escritor.cerrar()
    except BaseException:
        escritor.abortar()
        raise

    def fila_cruce(ahora: bool) -> dict:
        n = cruce[(ahora, True)] + cruce[(ahora, False)]
        return {"filas": n, "porcentaje_llena_60": round(cruce[(ahora, True)] / n * 100, 2) if n else None}

    resumen = {
        **conteo,
        "porcentaje_positivos": round((cruce[(True, True)] + cruce[(False, True)]) / conteo["con_etiqueta"] * 100, 2),
        "llena_ahora": fila_cruce(True),
        "no_llena_ahora": fila_cruce(False),
    }
    ruta_salida.with_name("resumen_objetivo.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resumen


def formatear_resumen_objetivo(resumen: dict) -> str:
    def pct(valor: float) -> str:
        return f"{valor:.2f}".replace(".", ",") + " %"

    sin = resumen["sin_etiqueta_falta_foto"] + resumen["sin_etiqueta_falta_estacion"]
    llena, no_llena = resumen["llena_ahora"], resumen["no_llena_ahora"]
    return "\n".join([
        f"Filas de entrada:   {formatear_entero(resumen['filas_entrada']):>12}",
        f"Con etiqueta:       {formatear_entero(resumen['con_etiqueta']):>12}"
        f"  ({pct(resumen['con_etiqueta'] / resumen['filas_entrada'] * 100)})",
        f"Sin etiqueta:       {formatear_entero(sin):>12}",
        f"  falta la foto de t+60:          {formatear_entero(resumen['sin_etiqueta_falta_foto']):>10}",
        f"  la foto no trae la estación:    {formatear_entero(resumen['sin_etiqueta_falta_estacion']):>10}",
        "",
        f"Positivos (llena a t+60): {pct(resumen['porcentaje_positivos'])}",
        "",
        f"{'Ahora':<14}{'Filas':>14}{'% llena a t+60':>18}",
        f"{'llena':<14}{formatear_entero(llena['filas']):>14}{pct(llena['porcentaje_llena_60']):>18}",
        f"{'no llena':<14}{formatear_entero(no_llena['filas']):>14}{pct(no_llena['porcentaje_llena_60']):>18}",
        "",
        "Predecir «sigue como está» ya es una referencia fuerte: el modelo tiene que superarla,",
        "sobre todo anticipando cuándo se llena una estación que ahora no lo está.",
    ])
