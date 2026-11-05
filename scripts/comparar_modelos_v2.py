"""Elige entre los modelos de la v2 con la regla fijada en CLAUDE.md, solo con la validación.

Parte de las configuraciones elegidas dentro de cada modelo (R1, L0 = A5 de la
ablación y X1) y de sus predicciones de validación ya guardadas: no entrena nada.
El orden de sencillez es logística < LightGBM < XGBoost: el siguiente modelo
sustituye al actual solo si lo mejora de forma clara, y en caso de empate gana
el más sencillo. El tiempo de entrenamiento no interviene en la regla.

Las demás comparaciones (la otra semilla, los subgrupos y la base A0) se
documentan, pero no deciden.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from valenbisi.experimento import Experimento
from valenbisi.memoria import comprobar_memoria, evitar_suspension
from valenbisi.seleccion import es_clara

RAIZ = Path(__file__).resolve().parents[1]
PROCESADOS = RAIZ / "data" / "processed"
PREDICCIONES = PROCESADOS / "modelos_v2"
ABLACION = PROCESADOS / "ablacion_v2"

# Cada modelo, de más a menos sencillo, con su resumen y la configuración que debe salir elegida en él.
MODELOS = [("logistica", "resumen_logistica_v2.json", "R1"),
           ("lightgbm", "resumen_lightgbm_v2.json", "L0"),
           ("xgboost", "resumen_xgboost_v2.json", "X1")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=PROCESADOS / "valenbisi_features_v2.parquet")
    parser.add_argument("--salida", type=Path, default=PROCESADOS / "resumen_comparacion_modelos_v2.json")
    parser.add_argument("--carpeta-vacia", type=Path, default=PROCESADOS / "comparacion_modelos_v2",
                        help="carpetas que pide Experimento; aquí no se escribe ningún modelo ni predicción")
    parser.add_argument("--replicas", type=int, default=200)
    parser.add_argument("--minimo-libre-gb", type=float, default=1.5)
    args = parser.parse_args()
    evitar_suspension()
    comprobar_memoria(args.minimo_libre_gb)

    exp = Experimento(args.features, args.salida, args.carpeta_vacia, args.carpeta_vacia, replicas=args.replicas)
    rutas = {"R1": PREDICCIONES / "R1.npy", "L0": PREDICCIONES / "L0.npy", "X1": PREDICCIONES / "X1.npy",
             "X2": PREDICCIONES / "X2.npy", "L0_s43": PREDICCIONES / "L0_s43.npy",
             "X1_s43": PREDICCIONES / "X1_s43.npy", "A5": ABLACION / "A5.npy", "A0": ABLACION / "A0.npy"}
    for nombre, ruta in rutas.items():
        p = np.load(ruta).astype("float64")
        if len(p) != len(exp.y_val):
            raise ValueError(f"{nombre}: {len(p)} predicciones para {len(exp.y_val)} filas de validación")
        exp.predicciones[nombre] = p
    if not np.array_equal(exp.predicciones["A5"], exp.predicciones["L0"]):
        raise ValueError("A5 y L0 deberían ser el mismo modelo")

    # Control: X2 - X1 tiene que reproducir la cifra publicada en resumen_xgboost_v2.json.
    resumenes = {modelo: json.loads((PROCESADOS / archivo).read_text(encoding="utf-8"))
                 for modelo, archivo, _ in MODELOS}
    control = exp.comparar("X2", "X1")
    if control != resumenes["xgboost"]["comparaciones"]["X2 - X1 (todas)"]:
        print(f"  Control distinto de lo publicado (normal si --replicas no es 200): {control}", flush=True)

    ruidos, pasos = {}, []
    actual, *resto = MODELOS
    modelo_actual, config_actual = actual[0], actual[2]
    for modelo, _, config in MODELOS:
        d = resumenes[modelo]["decisiones"]
        if d["elegida"] != config:
            raise ValueError(f"{modelo}: se esperaba {config} y el resumen dice {d['elegida']}")
        ruidos[config] = d["ruido_semilla_elegida"]
    for modelo, _, config in resto:
        nombre = "A5" if config == "L0" else config
        base = "A5" if config_actual == "L0" else config_actual
        diferencia = exp.comparar(nombre, base)
        ruido = max(ruidos[config], ruidos[config_actual])
        clara = es_clara(diferencia, ruido)
        pasos.append({"nuevo": nombre, "actual": base, "diferencia": diferencia, "ruido_mayor": ruido,
                      "umbral": 2 * ruido, "clara": clara})
        if clara:
            modelo_actual, config_actual = modelo, config

    # Solo para documentar.
    exp.comparar("X1_s43", "L0_s43")
    exp.comparar("X1", "A0")
    exp.comparar("A5", "A0")
    exp.comparar("X1", "A5", "no_llena_ahora")
    exp.comparar("X1", "A5", "libres_1_6")

    medidas = {nombre: exp.medir(exp.predicciones[nombre])["todas"] for nombre in ["R1", "A5", "X1", "A0"]}
    exp.resumen["medidas"] = {k: {m: float(v) for m, v in d.items()} for k, d in medidas.items()}
    exp.resumen["decisiones"] = {"pasos": pasos, "modelo_elegido": modelo_actual,
                                 "configuracion_elegida": "A5" if config_actual == "L0" else config_actual,
                                 "arboles": resumenes[modelo_actual]["decisiones"].get("arboles_elegida")}
    exp.guardar()
    print(f"\nElegido: {modelo_actual} ({exp.resumen['decisiones']['configuracion_elegida']})", flush=True)


if __name__ == "__main__":
    main()
