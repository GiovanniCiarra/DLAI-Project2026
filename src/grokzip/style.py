"""Shared style of the figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GEN = "#1b5e5a"
NOGEN = "#b8562a"
MAIN = "#14504c"
LONG = "#4a9d8f"
NC_MEM = "#d68a3a"
NC_WD0 = "#a03030"
GRID = "#cfcfcf"
ACCENT = "#2b2b2b"

CONDITION_COLORS = {
    "main": MAIN,
    "long": LONG,
    "nc_mem": NC_MEM,
    "nc_wd0": NC_WD0,
}
CONDITION_LABELS = {
    "main": "grokking, 30% data",
    "long": "grokking, 25% data",
    "nc_mem": "memorising, wd 1.0",
    "nc_wd0": "memorising, wd 0.0",
}
CONDITION_SHORT = {
    "main": "grok\n30%",
    "long": "grok\n25%",
    "nc_mem": "mem\nwd 1.0",
    "nc_wd0": "mem\nwd 0.0",
}

FULL_WIDTH = 6.9 


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.6,
            "lines.linewidth": 1.3,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save(fig, stem: str) -> None:
    from pathlib import Path

    Path("figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{stem}.{ext}")
    plt.close(fig)
    print(f"  figures/{stem}.pdf  +  .png")