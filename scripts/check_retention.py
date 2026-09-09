"""

This script verifies that the separation between the three populations persists at 
0.80 and 0.95 as well, recalibrating R_crit from scratch for each threshold using the same rule.
"""

import argparse
import csv
from collections import defaultdict

from grokzip.detector import calibrate, detection_step, final_r_star


def carica(path: str) -> dict[str, dict[str, list[dict]]]:
    per_run = defaultdict(list)
    for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if row["r_star"] == "":
            continue
        per_run[row["run"]].append(
            {
                "step": int(row["step"]),
                "r_star": int(row["r_star"]),
                "test_acc": float(row["test_acc"]),
                "condition": row["condition"],
            }
        )
    per_cond = defaultdict(dict)
    for run, traj in per_run.items():
        per_cond[traj[0]["condition"]][run] = traj
    return per_cond


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", type=str, nargs="+",
        default=[
            "0.80:results/trajectories_r80.csv",
            "0.90:results/trajectories.csv",
            "0.95:results/trajectories_r95.csv",
        ],
    )
    args = parser.parse_args()

    print(f"{'ritenz.':>8}{'R_crit':>8}{'r* gen.':>18}{'r* nc_mem':>12}"
          f"{'r* nc_wd0':>12}{'rilev.':>9}{'falsi':>7}")
    print("-" * 74)

    for voce in args.files:
        etichetta, path = voce.split(":", 1)
        per_cond = carica(path)
        cal = calibrate(per_cond["main"])
        r_crit = cal["r_crit"]

        gen = list(per_cond["main"].values()) + list(per_cond["long"].values())
        ctrl = list(per_cond["nc_mem"].values()) + list(per_cond["nc_wd0"].values())

        finali_gen = sorted(final_r_star(t) for t in gen)
        f_mem = sorted(final_r_star(t) for t in per_cond["nc_mem"].values())
        f_wd0 = sorted(final_r_star(t) for t in per_cond["nc_wd0"].values())
        rilevate = sum(detection_step(t, r_crit) is not None for t in gen)
        falsi = sum(detection_step(t, r_crit) is not None for t in ctrl)

        print(
            f"{etichetta:>8}{r_crit:>8}"
            f"{f'{finali_gen[0]}-{finali_gen[-1]}':>18}"
            f"{f'{f_mem[0]}-{f_mem[-1]}':>12}"
            f"{f'{f_wd0[0]}-{f_wd0[-1]}':>12}"
            f"{f'{rilevate}/{len(gen)}':>9}{f'{falsi}/{len(ctrl)}':>7}"
        )


if __name__ == "__main__":
    main()