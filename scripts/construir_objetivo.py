"""Añade la etiqueta llena_60 a data/processed/valenbisi_limpio.parquet.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\construir_objetivo.py
"""

import argparse
import sys
import time
from pathlib import Path

from valenbisi.objetivo import construir_objetivo, formatear_resumen_objetivo

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limpio", type=Path, default=RAIZ / "data" / "processed" / "valenbisi_limpio.parquet")
    parser.add_argument("--salida", type=Path, default=RAIZ / "data" / "processed" / "valenbisi_objetivo.parquet")
    args = parser.parse_args()

    inicio = time.monotonic()

    def progreso(i: int) -> None:
        if i % 10 == 0:
            print(f"  {i} lotes ({time.monotonic() - inicio:.0f} s)", flush=True)

    print(f"Leyendo {args.limpio}", flush=True)
    resumen = construir_objetivo(args.limpio, args.salida, progreso=progreso)
    print()
    print(formatear_resumen_objetivo(resumen))
    print(f"\nGuardado en {args.salida} ({time.monotonic() - inicio:.0f} s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
