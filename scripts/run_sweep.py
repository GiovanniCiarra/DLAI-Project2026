"""
Compression sweep across all runs and all anchors.

Produces `results/sweep_anchors.csv`, which contains one row per combination
(run, anchor, operator, level).
"""

import argparse
import csv
import time
from pathlib import Path

import torch

from grokzip.anchors import generalises, pick_anchors
from grokzip.measure import Evaluator, load_config, read_manifest
from grokzip.sweep import sweep_state_dict
from grokzip.train import pick_device

DEFAULT_RUNS = (
    [f"main_seed{s}" for s in range(8)]
    + [f"long_seed{s}" for s in range(4)]
    + [f"nc_mem_seed{s}" for s in range(2)]
    + [f"nc_wd0_seed{s}" for s in range(2)]
)


def condition_of(run: str) -> str:
    return run.rsplit("_seed", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--out", type=str, default="results/sweep_anchors.csv")
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")
    print(f"run da elaborare: {len(args.runs)}\n")

    rows: list[dict] = []
    start = time.time()

    for run in args.runs:
        config = load_config(f"results/{run}_history.csv")
        manifest = read_manifest(f"results/{run}_manifest.csv")
        evaluator = Evaluator(config, device)
        anchors = pick_anchors(manifest)

        etichette = []
        for name, anchor in anchors.items():
            state_dict = torch.load(anchor["path"], map_location="cpu")
            for row in sweep_state_dict(state_dict, evaluator):
                rows.append(
                    {
                        "run": run,
                        "condition": condition_of(run),
                        "seed": config.seed,
                        "train_fraction": config.train_fraction,
                        "weight_decay": config.weight_decay,
                        "anchor": name,
                        "step": anchor["step"],
                        "anchor_test_acc": anchor["test_acc"],
                        "generalises": int(generalises(anchor)),
                        **row,
                    }
                )
            etichette.append(f"{name}@{anchor['step']}")
        print(f"  {run:<16} {', '.join(etichette)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} righe scritte in {args.out}")
    print(f"tempo totale: {(time.time() - start) / 60:.1f} minuti")


if __name__ == "__main__":
    main()