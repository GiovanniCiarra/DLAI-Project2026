"""
For each operator, identify the level of maximum separation 
and report the mean difference and standard deviation between seeds.
"""

import argparse
import csv
from collections import defaultdict
from statistics import mean, stdev


def carica(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [
            {
                **row,
                "level": float(row["level"]),
                "train_acc": float(row["train_acc"]),
                "generalises": int(row["generalises"]),
            }
            for row in csv.DictReader(handle)
            if row["operator"] != "none"
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/sweep_anchors.csv")
    args = parser.parse_args()
    rows = carica(args.csv)

    gruppi = defaultdict(lambda: {"gen": [], "nogen": []})
    for row in rows:
        if row["anchor"] == "early":
            continue
        chiave = (row["operator"], row["level"])
        gruppi[chiave]["gen" if row["generalises"] else "nogen"].append(row["train_acc"])

    print("Confronto: ancore che generalizzano vs ancore che non generalizzano")
    print("(escluse le ancore 'early')\n")

    for operatore in ("rank", "quantize", "prune"):
        livelli = sorted(k[1] for k in gruppi if k[0] == operatore)
        print(f"--- {operatore}")
        print(f"{'livello':>9} {'generalizza':>22} {'non generalizza':>22} {'divario':>9}")
        migliore = None
        for livello in livelli:
            dati = gruppi[(operatore, livello)]
            if len(dati["gen"]) < 2 or len(dati["nogen"]) < 2:
                continue
            m_gen, m_no = mean(dati["gen"]), mean(dati["nogen"])
            s_gen, s_no = stdev(dati["gen"]), stdev(dati["nogen"])
            divario = m_gen - m_no
            print(
                f"{livello:>9.2f} {m_gen:>11.4f} ± {s_gen:.4f} "
                f"{m_no:>11.4f} ± {s_no:.4f} {divario:>+9.4f}"
            )
            if migliore is None or divario > migliore[1]:
                migliore = (livello, divario, len(dati["gen"]), len(dati["nogen"]))
        if migliore:
            print(
                f"  massima separazione a livello {migliore[0]:g}: "
                f"{migliore[1]:+.4f}  (n_gen={migliore[2]}, n_nogen={migliore[3]})\n"
            )


if __name__ == "__main__":
    main()