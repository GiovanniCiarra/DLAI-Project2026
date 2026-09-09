"""

Detector sensitivity to the choice of R_crit.

For each threshold, it reports the number of generalizing runs detected, 
the number of false positives produced on controls, and the median lead time.
"""

import argparse
from collections import defaultdict
from statistics import median

from grokzip.detector import detection_step, final_r_star
from scripts.run_detector import carica


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/trajectories.csv")
    parser.add_argument("--from-r", type=int, default=10)
    parser.add_argument("--to-r", type=int, default=40)
    args = parser.parse_args()

    per_run = carica(args.csv)
    per_cond = defaultdict(dict)
    for run, traj in per_run.items():
        per_cond[traj[0]["condition"]][run] = traj

    generalizzanti = list(per_cond["main"].values()) + list(per_cond["long"].values())
    controlli = list(per_cond["nc_mem"].values()) + list(per_cond["nc_wd0"].values())

    print(f"{'R_crit':>7}{'rilevate':>10}{'falsi pos.':>12}{'lead mediano':>14}")
    print("-" * 43)
    for r_crit in range(args.from_r, args.to_r + 1):
        rilevate, ritardi = 0, []
        for traj in generalizzanti:
            t_det = detection_step(traj, r_crit)
            if t_det is not None:
                rilevate += 1
                ritardi.append(t_det - traj[0]["t_grok50"])
        falsi = sum(detection_step(t, r_crit) is not None for t in controlli)
        mediana = f"{median(ritardi):+.0f}" if ritardi else "-"
        marca = "  <-- perfetto" if rilevate == len(generalizzanti) and falsi == 0 else ""
        print(f"{r_crit:>7}{rilevate:>4}/{len(generalizzanti):<5}"
              f"{falsi:>5}/{len(controlli):<6}{mediana:>14}{marca}")


if __name__ == "__main__":
    main()