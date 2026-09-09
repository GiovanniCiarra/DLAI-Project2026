"""
For each run, show r* immediately after memorization, r* at the end of training, 
and — only for runs that grok — the value of r* around the time of grokking. 
This helps determine whether the signal has the expected form before constructing the detector.
"""

import argparse
import csv
from collections import defaultdict


def carica(path):
    per_run = defaultdict(list)
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r["r_star"] == "":
            continue
        per_run[r["run"]].append(
            {
                "step": int(r["step"]),
                "r_star": int(r["r_star"]),
                "test_acc": float(r["test_acc"]),
                "condition": r["condition"],
                "t_grok50": int(r["t_grok50"]) if r["t_grok50"] else None,
            }
        )
    return per_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/trajectories.csv")
    args = parser.parse_args()
    per_run = carica(args.csv)

    print(
        f"{'run':<15}{'r* iniz.':>9}{'r* min':>8}{'r* fin.':>9}"
        f"{'t_grok50':>10}{'r* a t_grok':>12}"
    )
    print("-" * 63)

    for run in sorted(per_run):
        righe = sorted(per_run[run], key=lambda r: r["step"])
        t_grok = righe[0]["t_grok50"]

        r_a_grok = "-"
        if t_grok is not None:
            vicini = [r for r in righe if r["step"] <= t_grok]
            if vicini:
                r_a_grok = str(vicini[-1]["r_star"])

        print(
            f"{run:<15}{righe[0]['r_star']:>9}"
            f"{min(r['r_star'] for r in righe):>8}"
            f"{righe[-1]['r_star']:>9}"
            f"{str(t_grok) if t_grok else '-':>10}{r_a_grok:>12}"
        )


if __name__ == "__main__":
    main()