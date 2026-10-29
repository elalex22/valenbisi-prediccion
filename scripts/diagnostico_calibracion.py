"""Diagnóstico barato: ¿el desfase de calibración en validación es del modelo o del periodo?

LightGBM se queda algo corto en la zona media en validación. La tabla por
anclajes libres no tiene nada que sobreajustar, así que si también se queda
corta, la causa es que validación se llena más que entrenamiento con el mismo
número de huecos, no el modelo. Sin reentrenar nada:

1. Proporción real de llena_60 por valor de ``libres`` en el resto de
   entrenamiento, en su último bloque (jun–ago 2024) y en validación.
2. Tabla por libres ajustada sin ese bloque: real ÷ predicho en el bloque y en
   validación. Si el bloque no reproduce el desfase de validación, un calibrador
   ajustado con él no lo corregiría.
3. Observado ÷ esperado por mes (filas con al menos un anclaje libre), con la
   tabla ajustada con todo entrenamiento: muestra cuánto varía de un mes a otro
   y si se repite entre años.

Solo lee entrenamiento y validación: la prueba no se carga. Guarda
data/processed/resumen_diagnostico_calibracion.json.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\diagnostico_calibracion.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from valenbisi.dataset import formatear_entero
from valenbisi.evaluacion import ZONA, limites
from valenbisi.memoria import comprobar_memoria
from valenbisi.objetivo import HORIZONTE
from valenbisi.referencias import ajustar_libres, predecir_libres

RAIZ = Path(__file__).resolve().parents[1]
COLUMNAS = ["slot_utc", "libres", "llena_60"]
INICIO_BLOQUE = pd.Timestamp("2024-06-01", tz=ZONA).tz_convert("UTC")
TRAMOS = {"1": (1, 1), "2-3": (2, 3), "4-6": (4, 6), "7+": (7, None)}


def real_entre_predicho(modelo: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    p = predecir_libres(modelo, df)
    filas = []
    for tramo, (minimo, maximo) in TRAMOS.items():
        m = df["libres"].between(minimo, maximo if maximo is not None else df["libres"].max()).to_numpy()
        predicho, real = p[m].mean(), df["llena_60"].to_numpy()[m].mean()
        filas.append({"libres": tramo, "filas": int(m.sum()), "predicho_pct": round(predicho * 100, 3),
                      "real_pct": round(real * 100, 3), "real_entre_predicho": round(real / predicho, 3)})
    return pd.DataFrame(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=RAIZ / "data" / "processed" / "valenbisi_features.parquet")
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5,
                        help="RAM libre del equipo por debajo de la cual el script se para")
    args = parser.parse_args()
    comprobar_memoria(args.minimo_libre_gb)

    entrenamiento = pq.read_table(args.features, columns=COLUMNAS, filters=limites("entrenamiento")).to_pandas()
    validacion = pq.read_table(args.features, columns=COLUMNAS, filters=limites("validacion")).to_pandas()
    comprobar_memoria(args.minimo_libre_gb)
    # El bloque es el final de entrenamiento; el resto deja el hueco de 60 min antes de él.
    resto = entrenamiento[entrenamiento["slot_utc"] < INICIO_BLOQUE - HORIZONTE]
    bloque = entrenamiento[entrenamiento["slot_utc"] >= INICIO_BLOQUE]

    # 1. Proporción real por valor de libres.
    proporcion = pd.DataFrame({nombre: d.groupby("libres")["llena_60"].mean() * 100 for nombre, d in
                               (("resto", resto), ("bloque_jun_ago_2024", bloque), ("validacion", validacion))})
    proporcion = proporcion.loc[0:8].round(2)

    # 2. Tabla ajustada sin el bloque, en el bloque y en validación.
    sin_bloque = ajustar_libres(resto)
    desfase = {"bloque_jun_ago_2024": real_entre_predicho(sin_bloque, bloque),
               "validacion": real_entre_predicho(sin_bloque, validacion)}

    # 3. Observado ÷ esperado por mes, en filas con al menos un anclaje libre.
    tabla = ajustar_libres(entrenamiento)
    todo = pd.concat([entrenamiento.assign(conjunto="entrenamiento"), validacion.assign(conjunto="validacion")])
    todo = todo[todo["libres"] >= 1]
    todo["p"] = predecir_libres(tabla, todo)
    mes = todo["slot_utc"].dt.tz_convert(ZONA).dt.strftime("%Y-%m").rename("mes")
    por_mes = todo.groupby([mes, "conjunto"]).agg(filas=("p", "size"), esperado=("p", "sum"),
                                                  observado=("llena_60", "sum")).reset_index()
    por_mes["observado_entre_esperado"] = (por_mes["observado"] / por_mes["esperado"]).round(3)

    resumen = {
        "filas": {"resto": len(resto), "bloque_jun_ago_2024": len(bloque), "validacion": len(validacion)},
        "proporcion_real_por_libres_pct": {int(k): v for k, v in proporcion.to_dict("index").items()},
        "tabla_sin_bloque_real_entre_predicho": {k: v.to_dict("records") for k, v in desfase.items()},
        "observado_entre_esperado_por_mes": por_mes[["mes", "conjunto", "filas", "observado_entre_esperado"]]
        .to_dict("records"),
    }
    salida = args.features.with_name("resumen_diagnostico_calibracion.json")
    salida.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Filas: resto {formatear_entero(len(resto))}, bloque jun–ago 2024 {formatear_entero(len(bloque))}, "
          f"validación {formatear_entero(len(validacion))}")
    print("\n1) Proporción real de llena a t+60 por anclajes libres (%)")
    print(proporcion.to_string())
    for nombre, d in desfase.items():
        print(f"\n2) Tabla ajustada sin el bloque, en {nombre}")
        print(d.to_string(index=False))
    print("\n3) Observado ÷ esperado por mes (filas con al menos 1 libre)")
    print(por_mes[["mes", "conjunto", "filas", "observado_entre_esperado"]].to_string(index=False))
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
