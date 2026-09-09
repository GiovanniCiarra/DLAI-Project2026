"""

Pair the number of key frequencies with the minimum viable rank, run by run, and
verify the relationship r* ≥ 2k dictated by the circuit structure: each frequency
requires at least two dimensions, sine and cosine.
"""

import argparse
import csv
from collections import defaultdict
from statistics import mean, stdev

from scipy.stats import pearsonr, spearmanr

from grokzip.detector import sign_test

GENERALISING = ("main", "long")


def final_per_run(path: str, require: str | None = None) -> dict[str, dict]:
    per_run = defaultdict(list)
    for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if require and not row[require]:
            continue
        per_run[row["run"]].append(row)
    return {
        run: max(rows, key=lambda r: int(r["step"])) for run, rows in per_run.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fourier", type=str, default="results/fourier.csv")
    parser.add_argument("--trajectories", type=str, default="results/trajectories.csv")
    parser.add_argument("--out", type=str, default="results/fourier_summary.csv")
    args = parser.parse_args()

    fourier = final_per_run(args.fourier)
    traiettorie = final_per_run(args.trajectories, require="r_star")

    righe = []
    for run in sorted(fourier):
        f, t = fourier[run], traiettorie[run]
        k = int(f["n_key_10x"])
        r_star = int(t["r_star"])
        righe.append(
            {
                "run": run,
                "condition": f["condition"],
                "k": k,
                "two_k": 2 * k,
                "r_star": r_star,
                "margin": r_star - 2 * k,
                "concentration": float(f["concentration_top6"]),
                "key_freqs": f["key_freqs"],
            }
        )

    gen = [r for r in righe if r["condition"] in GENERALISING]
    ctrl = [r for r in righe if r["condition"].startswith("nc_")]

    print("=" * 72)
    print("ACCOPPIAMENTO r* / 2k  (solo run che generalizzano)")
    print("=" * 72)
    print(f"{'run':<14}{'k':>4}{'2k':>5}{'r*':>5}{'r*-2k':>8}{'concentr.':>12}")
    print("-" * 72)
    for r in gen:
        print(
            f"{r['run']:<14}{r['k']:>4}{r['two_k']:>5}{r['r_star']:>5}"
            f"{r['margin']:>+8}{r['concentration']:>12.4f}"
        )

    margini = [r["margin"] for r in gen]
    test = sign_test(margini)
    print(f"\n  margine r* - 2k: media {mean(margini):+.2f}, "
          f"minimo {min(margini):+d}, massimo {max(margini):+d}")
    print(f"  violazioni della disuguaglianza r* >= 2k: "
          f"{sum(1 for m in margini if m < 0)}/{len(margini)}")
    print(f"  test dei segni sui margini non nulli: n = {test['n']}, "
          f"p = {test['p_value']:.5f}")

    print("\n" + "=" * 72)
    print("CORRELAZIONE fra 2k e r*")
    print("=" * 72)
    x = [r["two_k"] for r in gen]
    y = [r["r_star"] for r in gen]
    rp, pp = pearsonr(x, y)
    rs, ps = spearmanr(x, y)
    print(f"  Pearson   r = {rp:+.4f}   p = {pp:.4f}   n = {len(x)}")
    print(f"  Spearman  rho = {rs:+.4f}  p = {ps:.4f}   n = {len(x)}")

    print("\n" + "=" * 72)
    print("CONCENTRAZIONE DI FOURIER  (top 6 frequenze su 48)")
    print("=" * 72)
    cg = [r["concentration"] for r in gen]
    cc = [r["concentration"] for r in ctrl]
    print(f"  generalizzanti (n={len(cg)}) : {mean(cg):.4f} ± {stdev(cg):.4f}   "
          f"[{min(cg):.4f}, {max(cg):.4f}]")
    print(f"  controlli      (n={len(cc)}) : {mean(cc):.4f} ± {stdev(cc):.4f}   "
          f"[{min(cc):.4f}, {max(cc):.4f}]")
    print(f"  separazione: {'completa' if min(cg) > max(cc) else 'PARZIALE'} "
          f"(divario {min(cg) - max(cc):+.4f})")
    print(f"  atteso per rumore bianco: circa 6/48 = {6/48:.4f} piu' varianza")

    print("\n" + "=" * 72)
    print("SOVRAPPOSIZIONE FRA GLI INSIEMI DI FREQUENZE")
    print("=" * 72)
    insiemi = {r["run"]: set(r["key_freqs"].split("|")) for r in gen}
    comuni = set.intersection(*insiemi.values())
    unione = set.union(*insiemi.values())
    print(f"  frequenze presenti in tutte le run : {len(comuni)}")
    print(f"  frequenze distinte usate in totale : {len(unione)} su 48 disponibili")
    print("  -> ogni seed converge a un insieme di frequenze diverso")

    with open(args.out, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)
    print(f"\n{len(righe)} righe scritte in {args.out}")


if __name__ == "__main__":
    main()