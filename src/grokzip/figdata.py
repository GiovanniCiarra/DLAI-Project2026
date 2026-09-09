"""
Preparation of data for the figures.
"""

import csv
from collections import defaultdict
from statistics import mean, stdev

GENERALISING_CONDITIONS = ("main", "long")
CONTROL_CONDITIONS = ("nc_mem", "nc_wd0")


def read_sweep(path: str) -> list[dict]:
    """
    It reads the sweep, excluding the reference line and the `early` anchors.

    The `early` anchors are excluded: they are the only ones close to the initialization.
    """
    rows = []
    for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if row["operator"] == "none" or row["anchor"] == "early":
            continue
        rows.append(
            {
                "operator": row["operator"],
                "level": float(row["level"]),
                "train_acc": float(row["train_acc"]),
                "generalises": int(row["generalises"]),
            }
        )
    return rows


def aggregate_sweep(rows: list[dict]) -> dict[str, list[dict]]:
    """
    Mean and standard deviation by operator and level, for the two groups.

    Returns operator -> list sorted by level; each entry includes the means and
    standard deviations for the two groups and their respective sample sizes.   
    """
    gruppi = defaultdict(lambda: {"gen": [], "nogen": []})
    for row in rows:
        chiave = (row["operator"], row["level"])
        gruppi[chiave]["gen" if row["generalises"] else "nogen"].append(row["train_acc"])

    per_operatore = defaultdict(list)
    for (operatore, livello), dati in gruppi.items():
        if not dati["gen"] or not dati["nogen"]:
            continue
        per_operatore[operatore].append(
            {
                "level": livello,
                "gen_mean": mean(dati["gen"]),
                "gen_sd": stdev(dati["gen"]) if len(dati["gen"]) > 1 else 0.0,
                "nogen_mean": mean(dati["nogen"]),
                "nogen_sd": stdev(dati["nogen"]) if len(dati["nogen"]) > 1 else 0.0,
                "n_gen": len(dati["gen"]),
                "n_nogen": len(dati["nogen"]),
            }
        )

    for operatore in per_operatore:
        per_operatore[operatore].sort(key=lambda r: r["level"])
    return dict(per_operatore)


def read_trajectories(path: str) -> dict[str, dict]:
    """Run trajectories, including only the checkpoints where r* is defined."""
    per_run = defaultdict(list)
    meta = {}
    for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if row["r_star"] == "":
            continue
        per_run[row["run"]].append(
            {
                "step": int(row["step"]),
                "r_star": int(row["r_star"]),
                "test_acc": float(row["test_acc"]),
            }
        )
        meta[row["run"]] = {
            "condition": row["condition"],
            "t_grok50": int(row["t_grok50"]) if row["t_grok50"] else None,
        }

    return {
        run: {"points": sorted(pts, key=lambda p: p["step"]), **meta[run]}
        for run, pts in per_run.items()
    }


def detection_step(points: list[dict], r_crit: int) -> int | None:
    """
    First step with r* <= r_crit. Intentional duplicate of detector.detection_step,
    to avoid making the figures module dependent on the detector module.
    """
    for point in points:
        if point["r_star"] <= r_crit:
            return point["step"]
    return None


def sensitivity_curve(
    trajectories: dict[str, dict], r_range: range
) -> list[dict]:
    """Detections and false positives for each threshold value."""
    gen = [t for t in trajectories.values() if t["condition"] in GENERALISING_CONDITIONS]
    ctrl = [t for t in trajectories.values() if t["condition"] in CONTROL_CONDITIONS]

    curva = []
    for r_crit in r_range:
        rilevate = sum(detection_step(t["points"], r_crit) is not None for t in gen)
        falsi = sum(detection_step(t["points"], r_crit) is not None for t in ctrl)
        curva.append(
            {
                "r_crit": r_crit,
                "detected": rilevate,
                "false_positives": falsi,
                "n_gen": len(gen),
                "n_ctrl": len(ctrl),
                "perfect": rilevate == len(gen) and falsi == 0,
            }
        )
    return curva


def perfect_interval(curve: list[dict]) -> tuple[int, int] | None:
    """Endpoints of the threshold interval with perfect separation."""
    perfetti = [row["r_crit"] for row in curve if row["perfect"]]
    return (min(perfetti), max(perfetti)) if perfetti else None


def read_detector(path: str) -> list[dict]:
    """Lead time per run, sorted by ascending value."""
    rows = [
        {
            "run": row["run"],
            "condition": row["condition"],
            "delta_grok": int(row["delta_grok"]),
            "delta_relative": float(row["delta_relative"]),
            "plateau": int(row["plateau"]),
        }
        for row in csv.DictReader(open(path, newline="", encoding="utf-8"))
    ]
    return sorted(rows, key=lambda r: r["delta_grok"])


def read_fourier_summary(path: str) -> list[dict]:
    return [
        {
            "run": row["run"],
            "condition": row["condition"],
            "k": int(row["k"]),
            "two_k": int(row["two_k"]),
            "r_star": int(row["r_star"]),
            "concentration": float(row["concentration"]),
        }
        for row in csv.DictReader(open(path, newline="", encoding="utf-8"))
    ]


def spread_offsets(keys: list, width: float = 0.22) -> list[float]:
    """
    Horizontal shifts to separate points with identical coordinates.

    Points sharing the same key are distributed symmetrically
    around the original position. Without this, overlapping points would
    make it appear as though there are fewer observations than there actually are.
    """
    from collections import Counter

    conteggi = Counter(keys)
    visti = Counter()
    offsets = []
    for chiave in keys:
        n = conteggi[chiave]
        indice = visti[chiave]
        visti[chiave] += 1
        offsets.append(0.0 if n == 1 else width * (indice - (n - 1) / 2))
    return offsets