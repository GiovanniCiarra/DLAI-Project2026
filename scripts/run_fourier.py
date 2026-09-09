"""
Fourier analysis along trajectories.
Produces results/fourier.csv.
"""

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import torch

from grokzip.fourier import concentration, digit_embeddings, frequency_energy, key_frequencies
from grokzip.measure import load_config, read_manifest
from grokzip.train import grokking_summary, pick_device

DEFAULT_RUNS = (
    [f"main_seed{s}" for s in range(8)]
    + [f"long_seed{s}" for s in range(4)]
    + [f"nc_mem_seed{s}" for s in range(2)]
    + [f"nc_wd0_seed{s}" for s in range(2)]
)
RATIOS = [5.0, 10.0, 20.0]


def read_history(path):
    with open(path, newline="", encoding="utf-8") as h:
        return [
            {"step": int(r["step"]), "train_acc": float(r["train_acc"]),
             "test_acc": float(r["test_acc"])}
            for r in csv.DictReader(h)
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--out", type=str, default="results/fourier.csv")
    args = parser.parse_args()

    pick_device()  
    rows, finali = [], []

    for run in args.runs:
        config = load_config(f"results/{run}_history.csv")
        manifest = read_manifest(f"results/{run}_manifest.csv")
        summary = grokking_summary(read_history(f"results/{run}_history.csv"))

        for riga in manifest:
            sd = torch.load(riga["path"], map_location="cpu")
            energia = frequency_energy(digit_embeddings(sd, config.p))
            conteggi = {r: len(key_frequencies(energia, r)) for r in RATIOS}
            rows.append(
                {
                    "run": run,
                    "condition": run.rsplit("_seed", 1)[0],
                    "seed": config.seed,
                    "step": riga["step"],
                    "test_acc": riga["test_acc"],
                    "n_key_5x": conteggi[5.0],
                    "n_key_10x": conteggi[10.0],
                    "n_key_20x": conteggi[20.0],
                    "concentration_top6": concentration(energia, 6),
                    "key_freqs": "|".join(str(k) for k in key_frequencies(energia)),
                    "t_grok50": summary["t_grok50"] or "",
                }
            )

        ultima = rows[-1]
        finali.append(ultima)
        print(
            f"  {run:<14} frequenze chiave (10x): {ultima['n_key_10x']:>2} "
            f"-> 2k = {2 * ultima['n_key_10x']:>2} | "
            f"concentrazione {ultima['concentration_top6']:.4f} | "
            f"k = {ultima['key_freqs']}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    print("VERIFICA DELLA PREVISIONE 8.2")
    print("=" * 70)
    gen = [r for r in finali if r["condition"] in ("main", "long")]
    for etichetta, chiave in [("5x", "n_key_5x"), ("10x", "n_key_10x"), ("20x", "n_key_20x")]:
        valori = [r[chiave] for r in gen]
        print(
            f"  soglia {etichetta:>3}: k = {mean(valori):.2f} ± {stdev(valori):.2f} "
            f"-> 2k = {2 * mean(valori):.1f}   (r* misurato: 10-15)"
        )

    print("\n  controlli negativi:")
    for r in finali:
        if r["condition"].startswith("nc_"):
            print(
                f"    {r['run']:<14} k = {r['n_key_10x']:>2} | "
                f"concentrazione {r['concentration_top6']:.4f}"
            )

    print(f"\n{len(rows)} righe scritte in {args.out}")


if __name__ == "__main__":
    main()