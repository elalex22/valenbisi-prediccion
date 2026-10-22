"""Construye data/processed/ a partir de data/raw/valenbici-master.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\construir_dataset.py
"""

import argparse
import sys
import time
from pathlib import Path

from valenbisi.dataset import construir_dataset, formatear_resumen

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAIZ / "data" / "raw" / "valenbici-master")
    parser.add_argument("--salida", type=Path, default=RAIZ / "data" / "processed")
    args = parser.parse_args()

    inicio = time.monotonic()

    def progreso(i: int, total: int) -> None:
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} zips ({time.monotonic() - inicio:.0f} s)", flush=True)

    print(f"Leyendo {args.raw}", flush=True)
    resumen = construir_dataset(args.raw, args.salida, progreso)
    print()
    print(formatear_resumen(resumen))
    print(f"\nGuardado en {args.salida} ({time.monotonic() - inicio:.0f} s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
