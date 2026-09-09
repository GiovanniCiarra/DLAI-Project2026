"""
It calibrates R_crit using only the `main` runs, applies it unchanged to the 
`long` runs and negative controls, and reports lead times and statistics.
"""

import argparse
import csv
from collections import defaultdict
from statistics import mean, median, stdev

from grokzip.detector import (
    calibrate,
    detection_step,
    final_r_star,
    plateau_minimum,
    sign_test,
)

ONSET_TEST = 0.15


def carica(path: str) -> dict[str, list[dict]]:
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
                "t_grok50": int(row["t_grok50"]) if row["t_grok50"] else None,
            }
        )
    return per_run


def onset_step(trajectory: list[dict]) -> int | None:
    """Primo step in cui la test accuracy supera l'altopiano. Riferimento secondario."""
    for row in sorted(trajectory, key=lambda r: r["step"]):
        if row["test_acc"] > ONSET_TEST:
            return row["step"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/trajectories.csv")
    parser.add_argument("--out", type=str, default="results/detector.csv")
    args = parser.parse_args()

    per_run = carica(args.csv)
    per_cond = defaultdict(dict)
    for run, traj in per_run.items():
        per_cond[traj[0]["condition"]][run] = traj

    # --- calibration, on main only -------------------------------------------
    cal = calibrate(per_cond["main"])
    r_crit = cal["r_crit"]

    print("=" * 70)
    print("CALIBRAZIONE  (solo condizione `main`, 8 run)")
    print("=" * 70)
    print(f"  P = max r* finale ................. {cal['p_max']}")
    print(f"  N = min r* di altopiano ........... {cal['n_min']}")
    print(f"  classi separate su main ........... {'si' if cal['separati'] else 'NO'}")
    print(f"  R_crit = floor((P + N) / 2) ....... {r_crit}")

    # --- final separation, all conditions ------------------------------
    print("\n" + "=" * 70)
    print(f"SEPARAZIONE A FINE TRAINING  (R_crit = {r_crit})")
    print("=" * 70)
    print(f"{'condizione':<10}{'run':>4}{'r* finale':>26}{'rilevato':>12}")
    print("-" * 70)
    for cond in ("main", "long", "nc_mem", "nc_wd0"):
        finali = sorted(final_r_star(t) for t in per_cond[cond].values())
        rilevati = sum(
            detection_step(t, r_crit) is not None for t in per_cond[cond].values()
        )
        elenco = ", ".join(str(v) for v in finali)
        print(f"{cond:<10}{len(finali):>4}{elenco:>26}{rilevati}/{len(finali):>11}")

    generalizzanti = list(per_cond["main"].values()) + list(per_cond["long"].values())
    controlli = list(per_cond["nc_mem"].values()) + list(per_cond["nc_wd0"].values())
    margine_pos = min(r_crit - final_r_star(t) for t in generalizzanti)
    margine_neg = min(min(r["r_star"] for r in t) - r_crit for t in controlli)
    print(f"\n  margine minimo lato generalizzante ... {margine_pos}")
    print(f"  margine minimo lato controlli ....... {margine_neg}")
    print(f"  falsi positivi sui controlli ........ "
          f"{sum(detection_step(t, r_crit) is not None for t in controlli)}/{len(controlli)}")

    # --- lead time ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LEAD TIME  (negativo = il rivelatore anticipa)")
    print("=" * 70)
    print(f"{'run':<14}{'t_grok50':>10}{'t_onset':>9}{'t_detect':>10}"
          f"{'Δ vs grok':>11}{'Δ vs onset':>12}{'Δ/altopiano':>13}")
    print("-" * 70)

    righe = []
    for cond in ("main", "long"):
        for run in sorted(per_cond[cond]):
            traj = per_cond[cond][run]
            t_grok = traj[0]["t_grok50"]
            t_onset = onset_step(traj)
            t_det = detection_step(traj, r_crit)
            t_mem = min(r["step"] for r in traj)

            d_grok = None if t_det is None else t_det - t_grok
            d_onset = None if (t_det is None or t_onset is None) else t_det - t_onset
            altopiano = t_grok - t_mem
            d_rel = None if d_grok is None else d_grok / altopiano

            righe.append(
                {
                    "run": run, "condition": cond, "r_crit": r_crit,
                    "t_grok50": t_grok, "t_onset": t_onset, "t_detect": t_det,
                    "delta_grok": d_grok, "delta_onset": d_onset,
                    "delta_relative": d_rel, "plateau": altopiano,
                    "r_star_final": final_r_star(traj),
                    "r_star_at_grok": [r["r_star"] for r in sorted(traj, key=lambda x: x["step"]) if r["step"] <= t_grok][-1],
                }
            )
            print(
                f"{run:<14}{t_grok:>10}{str(t_onset):>9}{str(t_det):>10}"
                f"{'' if d_grok is None else f'{d_grok:+}':>11}"
                f"{'' if d_onset is None else f'{d_onset:+}':>12}"
                f"{'' if d_rel is None else f'{d_rel:+.3f}':>13}"
            )

    # --- inferenza ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INFERENZA  (test dei segni bilaterale esatto, disegno appaiato entro run)")
    print("=" * 70)

    for etichetta, chiave in [("vs t_grok50", "delta_grok"), ("vs t_onset", "delta_onset")]:
        valori = [r[chiave] for r in righe if r[chiave] is not None]
        if not valori:
            continue
        test = sign_test(valori)
        verso = "RITARDA" if test["positivi"] > test["negativi"] else "ANTICIPA"
        print(f"\n  {etichetta}   n = {test['n']}")
        print(f"    media {mean(valori):+.1f} step | mediana {median(valori):+.1f}"
              f"{f' | sd {stdev(valori):.1f}' if len(valori) > 1 else ''}")
        print(f"    positivi (ritardo) {test['positivi']} | negativi (anticipo) {test['negativi']}")
        print(f"    p = {test['p_value']:.6f}   ->  il rivelatore {verso}")

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(righe[0].keys()))
        writer.writeheader()
        writer.writerows(righe)
    print(f"\n{len(righe)} righe scritte in {args.out}")


if __name__ == "__main__":
    main()