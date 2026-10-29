"""Comprobaciones de FORMATO del clon de github.com/ceferra/valenbici (fase 0 de la v2).

Solo mira nombres de fichero, hashes, cabeceras y número de líneas: nunca
imprime filas de datos de los zips nuevos, que son la prueba v2.

1. Los zips anteriores a la frontera son idénticos (SHA-256) a los de valenbici-master.
2. Los zips nuevos tienen el formato esperado: nombres, cabecera, separador y fotos vacías.
3. Cobertura: último día con fotos, días sin fotos y días incompletos (por la
   fecha de cada foto, no la del zip).

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python scripts\\comprobar_clon.py
"""

import argparse
import hashlib
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
FRONTERA = date(2025, 9, 24)
ZIP = re.compile(r"^(\d{2})-(\d{2})-(\d{4})\.zip$")
CSV = re.compile(r"^valenbici_(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})\.csv$")


def fecha_zip(nombre: str) -> date | None:
    m = ZIP.match(nombre)
    return date(int(m[3]), int(m[2]), int(m[1])) if m else None


def zips(carpeta: Path) -> dict[date, Path]:
    return {fecha_zip(p.name): p for p in carpeta.iterdir() if fecha_zip(p.name)}


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def estructura(ruta: Path) -> dict[str, tuple[str, int]]:
    """{miembro: (cabecera, nº de líneas de datos)}. La cabecera es la única línea que se lee."""
    salida = {}
    with zipfile.ZipFile(ruta) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            with z.open(info) as f:
                lineas = f.read().decode("utf-8", errors="replace").splitlines()
            salida[info.filename] = (lineas[0].strip() if lineas else "", max(len(lineas) - 1, 0))
    return salida


def rangos(dias: list[date]) -> list[str]:
    """Días seguidos agrupados como «inicio..fin»."""
    salida, inicio = [], None
    for i, d in enumerate(dias):
        inicio = inicio or d
        if i + 1 == len(dias) or dias[i + 1] != d + timedelta(1):
            salida.append(f"{inicio}" if inicio == d else f"{inicio}..{d}")
            inicio = None
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", type=Path, default=RAIZ / "data" / "raw" / "valenbici-master")
    parser.add_argument("--clon", type=Path, default=RAIZ / "data" / "raw" / "valenbici-git")
    args = parser.parse_args()

    m, c = zips(args.master), zips(args.clon)
    print(f"master: {len(m)} zips ({min(m)} a {max(m)});  clon: {len(c)} zips ({min(c)} a {max(c)})")
    otros = sorted(p.name for p in args.clon.iterdir() if not fecha_zip(p.name) and p.name != ".git")
    print("Otros ficheros en el clon:", otros)

    # 1. Zips anteriores a la frontera: ¿idénticos?
    antiguos = sorted(d for d in set(m) | set(c) if d < FRONTERA)
    solo_master = [d for d in antiguos if d in m and d not in c]
    solo_clon = [d for d in antiguos if d in c and d not in m]
    distintos = [d for d in antiguos if d in m and d in c and sha256(m[d]) != sha256(c[d])]
    print(f"\n1) Antes del {FRONTERA}: {len(antiguos)} fechas; solo en master {solo_master}; "
          f"solo en clon {solo_clon}; con hash distinto: {len(distintos)}")
    for d in distintos:
        em, ec = estructura(m[d]), estructura(c[d])
        distinto_n = [k for k in set(em) & set(ec) if em[k][1] != ec[k][1]]
        print(f"   {d}: miembros solo master {sorted(set(em) - set(ec))[:5]}, "
              f"solo clon {sorted(set(ec) - set(em))[:5]}, mismo nombre y distinto nº de filas {distinto_n[:5]}")
    # El zip de la frontera, aparte.
    for nombre, carpeta in (("master", m), ("clon", c)):
        if FRONTERA in carpeta:
            print(f"   {FRONTERA} en {nombre}: {len(estructura(carpeta[FRONTERA]))} miembros, "
                  f"hash {sha256(carpeta[FRONTERA])[:12]}")

    # Cabecera de referencia (la de después del 22-05-2023), de un zip antiguo de master.
    referencia = Counter(h for h, _ in estructura(m[date(2025, 9, 1)]).values()).most_common(1)[0][0]
    print(f"\nCabecera de referencia (master, 01-09-2025): {referencia}")

    # 2. Zips nuevos: formato.
    nuevos = sorted(d for d in c if d >= FRONTERA)
    cabeceras, primer_dia_cabecera = Counter(), {}
    fotos_por_dia, vacias_por_dia = defaultdict(set), Counter()
    nombres_raros, miembros_no_csv = [], []
    zips_con_otras_fechas = 0
    for d in nuevos:
        otras = False
        for miembro, (cabecera, filas) in estructura(c[d]).items():
            nombre = Path(miembro).name
            k = CSV.match(nombre)
            if not k:
                (nombres_raros if nombre.endswith(".csv") else miembros_no_csv).append(f"{c[d].name}/{miembro}")
                continue
            momento = datetime(int(k[3]), int(k[2]), int(k[1]), int(k[4]), int(k[5]), int(k[6]))
            fotos_por_dia[momento.date()].add(momento)
            otras |= momento.date() != d
            cabeceras[cabecera] += 1
            primer_dia_cabecera.setdefault(cabecera, momento.date())
            if filas == 0:
                vacias_por_dia[momento.date()] += 1
            if ";" not in cabecera:
                nombres_raros.append(f"{c[d].name}/{miembro} (cabecera sin ';')")
        zips_con_otras_fechas += otras

    print(f"\n2) Desde el {FRONTERA}: {len(nuevos)} zips ({nuevos[0]} a {nuevos[-1]})")
    for cab, n in cabeceras.most_common():
        texto = "IGUAL a la de referencia" if cab == referencia else cab
        print(f"   cabecera ({n} fotos, desde {primer_dia_cabecera[cab]}): {texto}")
    print(f"   CSV con nombre inesperado o sin ';': {nombres_raros[:10]} ({len(nombres_raros)})")
    print(f"   miembros que no son CSV: {miembros_no_csv[:10]} ({len(miembros_no_csv)})")
    print(f"   zips que traen fotos de otra fecha: {zips_con_otras_fechas}")
    print(f"   fotos vacías (solo cabecera): {sum(vacias_por_dia.values())} en {len(vacias_por_dia)} días")

    # 3. Cobertura por fecha de las fotos (no del zip).
    ultimo = max(fotos_por_dia)
    todos = [FRONTERA + timedelta(n) for n in range((ultimo - FRONTERA).days + 1)]
    faltan = [d for d in todos if d not in fotos_por_dia]
    incompletos = {d: len(fotos_por_dia[d]) for d in todos if 0 < len(fotos_por_dia.get(d, ())) < 96}
    print(f"\n3) Último día con fotos: {ultimo} ({len(fotos_por_dia[ultimo])} fotos). "
          f"Días en el periodo: {len(todos)}")
    print(f"   días sin ninguna foto: {len(faltan)}  {rangos(faltan)}")
    print(f"   días con menos de 96 fotos: {len(incompletos)}")
    print("   " + ", ".join(f"{d}:{n}" for d, n in sorted(incompletos.items())))
    mas_de_96 = {str(d): len(v) for d, v in fotos_por_dia.items() if len(v) > 96}
    print(f"   fotos totales: {sum(len(v) for v in fotos_por_dia.values())}; días con más de 96: {mas_de_96}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
