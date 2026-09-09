"""
Reads the CSVs produced by the previous stages and generates the figures
"""

import argparse
from statistics import mean, median

import matplotlib.pyplot as plt

from grokzip.detector import sign_test
from grokzip.figdata import (
    aggregate_sweep,
    perfect_interval,
    read_detector,
    read_fourier_summary,
    read_sweep,
    read_trajectories,
    sensitivity_curve,
    spread_offsets,
)
from grokzip.style import (
    ACCENT,
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_SHORT,
    FULL_WIDTH,
    GEN,
    GRID,
    NOGEN,
    apply_style,
    save,
)

R_CRIT = 25  # calibrated solely on `main` runs


def figura1(path_sweep: str) -> None:
    dati = aggregate_sweep(read_sweep(path_sweep))
    specifiche = [
        ("rank", "truncation rank", "SVD rank truncation", "log", False),
        ("prune", "sparsity", "magnitude pruning", "linear", True),
        ("quantize", "weight bits", "uniform quantisation", "linear", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.25), sharey=True)

    for ax, (operatore, xlabel, titolo, scala, inverti) in zip(axes, specifiche):
        righe = dati[operatore]
        x = [r["level"] for r in righe]

        for chiave, colore, nome in [
            ("gen", GEN, "generalising (n=12)"),
            ("nogen", NOGEN, "memorising (n=16)"),
        ]:
            media = [r[f"{chiave}_mean"] for r in righe]
            sd = [r[f"{chiave}_sd"] for r in righe]
            ax.plot(x, media, color=colore, label=nome, marker="o", markersize=2.2)
            ax.fill_between(
                x,
                [m - s for m, s in zip(media, sd)],
                [m + s for m, s in zip(media, sd)],
                color=colore, alpha=0.16, linewidth=0,
            )

        livello_riportato = {"rank": 16.0, "prune": 0.40, "quantize": 3.0}[operatore]
        voce = next(r for r in righe if r["level"] == livello_riportato)
        divario = voce["gen_mean"] - voce["nogen_mean"]
        ax.axvline(voce["level"], color=ACCENT, linestyle=":", linewidth=0.8)

        if scala == "log":
            ax.set_xscale("log", base=2)
            ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
            ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128"])
        if inverti:
            ax.invert_xaxis()

        ax.set_xlabel(xlabel)
        ax.set_title(titolo, pad=4)
        ax.text(
            0.97, 0.06, f"gap {divario:+.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color=ACCENT,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=GRID, linewidth=0.5),
        )

    axes[0].set_ylabel("train accuracy")
    axes[0].set_ylim(-0.04, 1.08)
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.4)

    save(fig, "fig1_operators")


def figura2(path_traj: str) -> None:
    """Traiettorie, con striscia laterale dei valori finali di r*."""
    traiettorie = read_trajectories(path_traj)

    fig = plt.figure(figsize=(FULL_WIDTH, 3.1))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[7, 1], height_ratios=[1, 1.75],
        hspace=0.10, wspace=0.04,
    )
    ax_alto = fig.add_subplot(gs[0, 0])
    ax_basso = fig.add_subplot(gs[1, 0], sharex=ax_alto)
    ax_strip = fig.add_subplot(gs[1, 1], sharey=ax_basso)

    viste = set()
    for run in sorted(traiettorie):
        traj = traiettorie[run]
        cond = traj["condition"]
        colore = CONDITION_COLORS[cond]
        nome = None
        if cond not in viste:
            nome = CONDITION_LABELS[cond]
            viste.add(cond)
        step = [p["step"] for p in traj["points"]]
        ax_alto.plot(step, [p["test_acc"] for p in traj["points"]],
                     color=colore, alpha=0.8, linewidth=0.85)
        ax_basso.plot(step, [p["r_star"] for p in traj["points"]],
                      color=colore, alpha=0.8, linewidth=0.85, label=nome)

    ax_alto.set_ylabel("test\naccuracy")
    ax_alto.set_ylim(-0.06, 1.08)
    ax_alto.tick_params(labelbottom=False)

    ax_basso.axhline(R_CRIT, color=ACCENT, linestyle="--", linewidth=1.0, zorder=1)
    ax_basso.set_xscale("log")
    ax_basso.set_xlabel("optimisation step")
    ax_basso.set_ylabel("minimal viable rank  $r^*$")
    ax_basso.set_ylim(0, 95)
    ax_basso.legend(loc="upper right", frameon=False, ncol=2,
                handlelength=1.4, columnspacing=1.0,
                bbox_to_anchor=(1.0, 1.02))

    ordine = ["main", "long", "nc_mem", "nc_wd0"]
    posizioni = {c: i for i, c in enumerate(ordine)}
    finali = [
        (traj["condition"], traj["points"][-1]["r_star"])
        for traj in traiettorie.values()
    ]
    offsets = spread_offsets([(c, r) for c, r in finali], width=0.30)
    for (cond, r_star), dx in zip(finali, offsets):
        ax_strip.scatter(posizioni[cond] + dx, r_star,
                         color=CONDITION_COLORS[cond], s=20, zorder=3,
                         edgecolor="white", linewidth=0.4)

    ax_strip.axhline(R_CRIT, color=ACCENT, linestyle="--", linewidth=1.0)
    ax_strip.text(3.6, R_CRIT + 2, f"$R_{{crit}}={R_CRIT}$",
                  fontsize=7, color=ACCENT, ha="right", va="bottom")
    ax_strip.set_xlim(-0.7, 3.7)
    ax_strip.set_xticks([])
    ax_strip.tick_params(labelleft=False)
    ax_strip.set_xlabel("final $r^*$", fontsize=7)
    ax_strip.grid(axis="x", visible=False)
    save(fig, "fig2_trajectories")


def figura3(path_traj: str, path_det: str) -> None:
    """Robustezza della soglia e lead time."""
    traiettorie = read_trajectories(path_traj)
    curva = sensitivity_curve(traiettorie, range(8, 45))
    intervallo = perfect_interval(curva)
    lead = read_detector(path_det)

    fig, (ax_sx, ax_dx) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, 2.5), gridspec_kw={"width_ratios": [1, 1.3]}
    )

    x = [r["r_crit"] for r in curva]
    if intervallo:
        ax_sx.axvspan(intervallo[0], intervallo[1], color=GEN, alpha=0.10, linewidth=0)
        ax_sx.annotate(
            f"perfect separation\n$R_{{crit}} \\in [{intervallo[0]}, {intervallo[1]}]$",
            xy=((intervallo[0] + intervallo[1]) / 2, 0),
            xytext=((intervallo[0] + intervallo[1]) / 2, 4.6),
            ha="center", fontsize=6.8, color=ACCENT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GRID, linewidth=0.5),
        )
    ax_sx.plot(x, [r["detected"] for r in curva], color=GEN,
               label="runs detected (of 12)", marker="o", markersize=2)
    ax_sx.plot(x, [r["false_positives"] for r in curva], color=NOGEN,
               label="false positives (of 4)", marker="s", markersize=2)
    ax_sx.axvline(R_CRIT, color=ACCENT, linestyle="--", linewidth=0.9)
    ax_sx.text(R_CRIT + 0.7, 8.0, "calibrated", fontsize=6, color=ACCENT, rotation=90)
    ax_sx.set_xlabel("detection threshold  $R_{crit}$")
    ax_sx.set_ylabel("number of runs")
    ax_sx.set_ylim(-0.6, 15.2)
    ax_sx.legend(loc="upper left", frameon=False, handlelength=1.4,
                 bbox_to_anchor=(0.0, 1.02))

    y = list(range(len(lead)))
    colori = [CONDITION_COLORS[r["condition"]] for r in lead]
    ax_dx.barh(y, [r["delta_grok"] for r in lead], color=colori, height=0.68)
    ax_dx.axvline(0, color=ACCENT, linewidth=1.0)
    ax_dx.set_yticks(y)
    ax_dx.set_yticklabels([r["run"].replace("_seed", " s") for r in lead], fontsize=6)
    ax_dx.set_xlabel("lead time  $t_{detect} - t_{grok}$  (steps)")
    ax_dx.set_ylim(-0.8, len(lead) - 0.2)

    valori = [r["delta_grok"] for r in lead]
    test = sign_test(valori)
    ax_dx.text(0.02, 0.06, "detector earlier  $\\leftarrow$",
               transform=ax_dx.transAxes, fontsize=6.5, color=ACCENT)
    ax_dx.text(0.98, 0.06, "$\\rightarrow$  later", ha="right",
               transform=ax_dx.transAxes, fontsize=6.5, color=ACCENT)
    ax_dx.set_title(
        f"median {median(valori):+.0f} steps · {test['negativi']}/{test['n']} earlier"
        f" · sign test $p = {test['p_value']:.3f}$  (n.s.)",
        fontsize=7.5, pad=4,
    )
    save(fig, "fig3_detector")


def figura4(path_fourier: str) -> None:
    """Consistenza fra rango e frequenze, e concentrazione spettrale."""
    righe = read_fourier_summary(path_fourier)
    gen = [r for r in righe if r["condition"] in ("main", "long")]

    fig, (ax_sx, ax_dx) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))

    limiti = [4, 17]
    ax_sx.fill_between(limiti, limiti, [limiti[1]] * 2, color=GEN, alpha=0.07, linewidth=0)
    ax_sx.plot(limiti, limiti, color=ACCENT, linestyle="--", linewidth=0.9)
    ax_sx.text(13.4, 12.1, "$r^* = 2k$", fontsize=7, color=ACCENT, rotation=41)

    coppie = [(r["two_k"], r["r_star"]) for r in gen]
    offsets = spread_offsets(coppie, width=0.34)
    for r, dx in zip(gen, offsets):
        ax_sx.scatter(r["two_k"] + dx, r["r_star"],
                      color=CONDITION_COLORS[r["condition"]], s=30, zorder=3,
                      edgecolor="white", linewidth=0.6)

    ax_sx.set_xlabel("$2k$   (key Fourier frequencies $\\times$ 2)")
    ax_sx.set_ylabel("minimal viable rank  $r^*$")
    ax_sx.set_xlim(*limiti)
    ax_sx.set_ylim(*limiti)
    ax_sx.set_xticks([4, 6, 8, 10, 12, 14, 16])
    ax_sx.set_yticks([4, 6, 8, 10, 12, 14, 16])
    ax_sx.set_title("$r^* \\geq 2k$ holds in 12/12 runs", fontsize=8, pad=4)
    ax_sx.text(0.03, 0.94, "points nudged sideways\nwhere runs coincide",
               transform=ax_sx.transAxes, fontsize=6, color=ACCENT,
               va="top", style="italic")

    ordine = ["main", "long", "nc_mem", "nc_wd0"]
    for i, cond in enumerate(ordine):
        valori = [r["concentration"] for r in righe if r["condition"] == cond]
        dx = spread_offsets([round(v, 2) for v in valori], width=0.20)
        for v, off in zip(valori, dx):
            ax_dx.scatter(i + off, v, color=CONDITION_COLORS[cond], s=30, zorder=3,
                          edgecolor="white", linewidth=0.6)

    ax_dx.axhline(6 / 48, color=ACCENT, linestyle=":", linewidth=0.9)
    ax_dx.text(-0.42, 6 / 48 + 0.025, "white-noise level",
               fontsize=6.5, color=ACCENT, ha="left")
    ax_dx.set_xticks(range(len(ordine)))
    ax_dx.set_xticklabels([CONDITION_SHORT[c] for c in ordine], fontsize=6.5)
    ax_dx.set_xlim(-0.6, 3.6)
    ax_dx.set_ylabel("energy in top-6 of 48 frequencies")
    ax_dx.set_ylim(0, 1.08)
    ax_dx.set_title("spectral concentration", fontsize=8, pad=4)
    ax_dx.grid(axis="x", visible=False)
    save(fig, "fig4_fourier")


def numeri_chiave(path_traj: str, path_det: str, path_fourier: str) -> None:
    traiettorie = read_trajectories(path_traj)
    intervallo = perfect_interval(sensitivity_curve(traiettorie, range(5, 60)))
    lead = read_detector(path_det)
    valori = [r["delta_grok"] for r in lead]
    test = sign_test(valori)
    fourier = read_fourier_summary(path_fourier)
    margini = [r["r_star"] - r["two_k"] for r in fourier
               if r["condition"] in ("main", "long")]

    print("\n" + "=" * 62)
    print("NUMERI DA CITARE NEL TESTO")
    print("=" * 62)
    print(f"  R_crit calibrata ................. {R_CRIT}")
    print(f"  intervallo di soglie perfette .... [{intervallo[0]}, {intervallo[1]}]")
    print(f"  checkpoint totali analizzati ..... "
          f"{sum(len(t['points']) for t in traiettorie.values())}")
    print(f"  lead time mediano ................ {median(valori):+.1f} step")
    print(f"  lead time medio .................. {mean(valori):+.1f} step")
    print(f"  anticipi / totale ................ {test['negativi']}/{test['n']}")
    print(f"  p del test dei segni ............. {test['p_value']:.3f}")
    print(f"  margine r*-2k medio .............. {mean(margini):+.2f}")
    print(f"  violazioni di r* >= 2k ........... "
          f"{sum(1 for m in margini if m < 0)}/{len(margini)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="results/sweep_anchors.csv")
    parser.add_argument("--trajectories", default="results/trajectories.csv")
    parser.add_argument("--detector", default="results/detector.csv")
    parser.add_argument("--fourier", default="results/fourier_summary.csv")
    args = parser.parse_args()

    apply_style()
    print("generazione delle figure:")
    figura1(args.sweep)
    figura2(args.trajectories)
    figura3(args.trajectories, args.detector)
    figura4(args.fourier)
    numeri_chiave(args.trajectories, args.detector, args.fourier)


if __name__ == "__main__":
    main()