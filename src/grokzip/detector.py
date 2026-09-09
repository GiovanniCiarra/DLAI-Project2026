"""
Generalization detector based on minimum viable rank.

The detector triggers at the first step where r*(t) <= R_crit. Once R_crit is set,
it requires no test set: r* is a function solely of the training data.

The threshold is calibrated on a reference condition (`main`)—where test labels
are available—and applied without modification to held-out conditions.
The labels are used only once, during calibration, not at every point of use.
"""

import math

PLATEAU_TEST_MAX = 0.15  # same threshold used in anchors.py, fixed before the data


def detection_step(trajectory: list[dict], r_crit: int) -> int | None:
    """
    First step where r* <= r_crit. None if this never occurs.

    `trajectory` is a list of dictionaries containing at least the keys `step` and `r_star`,
    limited to the checkpoints where r* is defined.
    """
    for row in sorted(trajectory, key=lambda r: r["step"]):
        if row["r_star"] <= r_crit:
            return row["step"]
    return None


def plateau_minimum(trajectory: list[dict]) -> int | None:
    """
    Minimum r* among the checkpoints saved without generalizing.

    It is the negative representative of the calibration: the lowest value reached
    by a model that has not yet generalized.
    """
    valori = [
        row["r_star"] for row in trajectory if row["test_acc"] <= PLATEAU_TEST_MAX
    ]
    return min(valori) if valori else None


def final_r_star(trajectory: list[dict]) -> int:
    """r* at the last checkpoint of the trajectory."""
    return max(trajectory, key=lambda r: r["step"])["r_star"]


def calibrate(trajectories: dict[str, list[dict]]) -> dict:
    """
    Calibrate R_crit based solely on the reference condition.

    Rule established before observing the detector results:

        P = maximum final r* across all runs        (examples that generalize)
        N = minimum plateau-bottom value across all runs (examples that do not generalize)
        R_crit = floor((P + N) / 2)

    The midpoint is chosen because it maximizes the minimum margin between the two classes
    without requiring any data-based optimization: it is not a value selected to
    maximize performance, but rather the center of the separation interval.
    """
    positivi = [final_r_star(t) for t in trajectories.values()]
    negativi = [
        m for t in trajectories.values() if (m := plateau_minimum(t)) is not None
    ]
    if not negativi:
        raise RuntimeError("nessun checkpoint di altopiano nella condizione di calibrazione")

    p_max, n_min = max(positivi), min(negativi)
    return {
        "r_crit": (p_max + n_min) // 2,
        "p_max": p_max,
        "n_min": n_min,
        "separati": p_max < n_min,
        "n_run": len(trajectories),
    }


def sign_test(differences: list[float]) -> dict:
    """
    Exact two-sided sign test. Zero differences are excluded.
    """
    positivi = sum(1 for d in differences if d > 0)
    negativi = sum(1 for d in differences if d < 0)
    n = positivi + negativi
    if n == 0:
        return {"n": 0, "positivi": 0, "negativi": 0, "p_value": 1.0}

    k = min(positivi, negativi)
    coda = sum(math.comb(n, i) for i in range(k + 1))
    return {
        "n": n,
        "positivi": positivi,
        "negativi": negativi,
        "p_value": min(1.0, 2 * coda / 2**n),
    }