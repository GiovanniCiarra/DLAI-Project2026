"""
Produces results/trajectories.csv, with one row per checkpoint.

r*(t) is calculated using only the training data. Test accuracy appears in the
same row but plays no part in the calculation; it serves solely to validate
the detector post hoc.
"""

import argparse
import csv
import time
from pathlib import Path

import torch

from grokzip.measure import Evaluator, load_config, read_manifest
from grokzip.threshold import minimal_rank
from grokzip.train import grokking_summary, pick_device

DEFAULT_RUNS = (
    [f"main_seed{s}" for s in range(8)]
    + [f"long_seed{s}" for s in range(4)]
    + [f"nc_mem_seed{s}" for s in range(2)]
    + [f"nc_wd0_seed{s}" for s in range(2)]
)


def read_history(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [
            {
                "step": int(r["step"]),
                "train_acc": float(r["train_acc"]),
                "test_acc": float(r["test_acc"]),
            }
            for r in csv.DictReader(handle)
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--retention", type=float, default=0.9)
    parser.add_argument("--out", type=str, default="results/trajectories.csv")
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device} | ritenzione {args.retention}\n")

    rows: list[dict] = []
    start = time.time()

    for run in args.runs:
        config = load_config(f"results/{run}_history.csv")
        manifest = read_manifest(f"results/{run}_manifest.csv")
        evaluator = Evaluator(config, device)

        summary = grokking_summary(read_history(f"results/{run}_history.csv"))

        definiti = 0
        for riga in manifest:
            sd = torch.load(riga["path"], map_location="cpu")
            r_star, baseline = minimal_rank(sd, evaluator, retention=args.retention)
            if r_star is not None:
                definiti += 1
            rows.append(
                {
                    "run": run,
                    "condition": run.rsplit("_seed", 1)[0],
                    "seed": config.seed,
                    "train_fraction": config.train_fraction,
                    "weight_decay": config.weight_decay,
                    "step": riga["step"],
                    "r_star": "" if r_star is None else r_star,
                    "train_acc": baseline,
                    "test_acc": riga["test_acc"],
                    "param_norm": riga["param_norm"],
                    "t_mem": summary["t_mem"] if summary["t_mem"] is not None else "",
                    "t_grok50": summary["t_grok50"] if summary["t_grok50"] is not None else "",
                }
            )

        finali = [r for r in rows if r["run"] == run and r["r_star"] != ""]
        print(
            f"  {run:<14} {definiti}/{len(manifest)} checkpoint con r* definito "
            f"| r* iniziale {finali[0]['r_star']:>3} -> finale {finali[-1]['r_star']:>3}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} righe scritte in {args.out}")
    print(f"tempo totale: {(time.time() - start) / 60:.1f} minuti")


if __name__ == "__main__":
    main()