"""Estilo común de las figuras: colores fijos por modelo y ejes discretos."""

from pathlib import Path

# Paleta categórica por defecto, en orden fijo (azul, naranja, aguamarina): cada
# modelo conserva su color en todas las figuras. Gris para las líneas de referencia.
COLORES = {"persistencia": "#2a78d6", "frecuencia": "#eb6834", "libres": "#8a4fc7", "lightgbm": "#1baf7a"}
GRIS, TEXTO, TEXTO_SUAVE = "#8f8e8a", "#0b0b0b", "#52514e"


def estilo_ejes(ax) -> None:
    ax.grid(color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)
    ax.tick_params(colors=TEXTO_SUAVE)


def guardar(fig, ruta: Path) -> None:
    import matplotlib.pyplot as plt

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
