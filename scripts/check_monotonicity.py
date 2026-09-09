"""
The binary search used for r* is exact only if the set of admissible 
ranks is an upper interval. This script measures this on the full grid, 
using a sample of checkpoints selected to cover all conditions and training phases.
"""

import argparse

import torch

from grokzip.measure import Evaluator, load_config, read_manifest
from grokzip.threshold import MAX_RANK, MIN_TRAIN_ACC, monotonicity_violations, retention_curve
from grokzip.train import pick_device

RUNS = ["main_seed0", "main_seed3", "long_seed1", "nc_mem_seed0", "nc_wd0_seed0"]
RETENTION = 0.9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, nargs="+", default=RUNS)
    parser.add_argument("--per-run", type=int, default=5)
    args = parser.parse_args()

    device = pick_device()
    ranks = list(range(1, MAX_RANK + 1))
    totale, violazioni_totali = 0, 0

    print(f"griglia completa: ranghi 1..{MAX_RANK} | soglia di ritenzione {RETENTION}\n")

    for run in args.runs:
        config = load_config(f"results/{run}_history.csv")
        manifest = read_manifest(f"results/{run}_manifest.csv")
        evaluator = Evaluator(config, device)

        validi = [r for r in manifest if r["train_acc"] >= MIN_TRAIN_ACC]
        passo = max(1, len(validi) // args.per_run)
        campione = validi[::passo][: args.per_run]

        for riga in campione:
            sd = torch.load(riga["path"], map_location="cpu")
            baseline, curva = retention_curve(sd, evaluator, ranks)
            violazioni = monotonicity_violations(curva, RETENTION)
            ammissibili = [r for r in ranks if curva[r] >= RETENTION]
            r_star = min(ammissibili) if ammissibili else None

            totale += 1
            violazioni_totali += violazioni
            print(
                f"  {run:<14} step {riga['step']:>6} | r* {str(r_star):>4} "
                f"| violazioni {violazioni:>3} | test {riga['test_acc']:.4f}"
            )

    print(f"\ncheckpoint esaminati: {totale}")
    print(f"violazioni di monotonia totali: {violazioni_totali}")
    if violazioni_totali == 0:
        print("La ricerca binaria e' esatta su questo campione.")
    else:
        print(
            "Monotonia non perfetta: r* da ricerca binaria puo' sovrastimare di poco.\n"
            "Riportare il tasso di violazione nel report."
        )


if __name__ == "__main__":
    main()