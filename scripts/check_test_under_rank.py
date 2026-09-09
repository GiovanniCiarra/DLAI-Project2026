"""
Check whether rank truncation also preserves test accuracy.

Robustness check: if the truncated model were to retain training accuracy
but lose test accuracy, r* would not be tracking the dimensionality of the generalizing circuit,
but merely the residual capacity to fit the examples already seen.
"""

import csv
from statistics import mean, stdev

rows = list(csv.DictReader(open("results/sweep_anchors.csv", encoding="utf-8")))
gen = [r for r in rows if r["operator"] == "rank" and int(r["generalises"]) == 1]

print(f"{'rank':>6}{'train':>18}{'test':>18}")
print("-" * 42)
for level in sorted({float(r["level"]) for r in gen}):
    at = [r for r in gen if float(r["level"]) == level]
    tr = [float(r["train_acc"]) for r in at]
    te = [float(r["test_acc"]) for r in at]
    print(
        f"{int(level):>6}{mean(tr):>10.4f} ± {stdev(tr):.4f}"
        f"{mean(te):>10.4f} ± {stdev(te):.4f}"
    )