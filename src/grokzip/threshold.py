"""
Minimum vital rank r*: the compression threshold.

r*(theta) is the minimum rank at which the model theta—truncated via SVD—retains
at least a `retention` fraction of its training accuracy.

This measure relies exclusively on training data. It is the project's core property:
it distinguishes memorization from generalization without involving any test set
in the calculation.
"""

from grokzip.compression import truncate_rank

MAX_RANK = 128  # full rank for every model tensor: the truncation is the identity
MIN_TRAIN_ACC = 0.99  # Below this threshold, r* is undefined (there is nothing to preserve).


def retention_curve(state_dict, evaluator, ranks) -> tuple[float, dict[int, float]]:
    """
    Retention of training accuracy at each grid rank.

    Returns the uncompressed accuracy and the rank-to-retention map.
    Used for monotonicity verification; the mass calculation uses minimal_rank.
    """
    baseline = evaluator.train_accuracy(state_dict)
    if baseline == 0:
        return 0.0, {r: 0.0 for r in ranks}
    curve = {
        r: evaluator.train_accuracy(truncate_rank(state_dict, r)) / baseline
        for r in ranks
    }
    return baseline, curve


def monotonicity_violations(curve: dict[int, float], retention: float) -> int:
    """
    The number of times the curve drops below the threshold after having exceeded it.

    If this value is zero, the set of admissible ranks is an upper interval, and
    binary search is exact.
    """
    ranks = sorted(curve)
    violations, visto_ammissibile = 0, False
    for rank in ranks:
        if curve[rank] >= retention:
            visto_ammissibile = True
        elif visto_ammissibile:
            violations += 1
    return violations


def minimal_rank(
    state_dict,
    evaluator,
    retention: float = 0.9,
    max_rank: int = MAX_RANK,
) -> tuple[int | None, float]:
    """
    Minimum rank with retention >= `retention`, found via binary search.

    Returns (r*, uncompressed training accuracy). r* is None when the
    checkpoint has not yet stored a model—i.e., when the uncompressed accuracy is
    below MIN_TRAIN_ACC—because in that case there is no learned function to preserve,
    and relative retention would not be interpretable.

    The maximum rank is always valid by construction (full-rank truncation is the
    identity operation), so the search always terminates.
    """
    baseline = evaluator.train_accuracy(state_dict)
    if baseline < MIN_TRAIN_ACC:
        return None, baseline

    lo, hi = 1, max_rank
    while lo < hi:
        mid = (lo + hi) // 2
        acc = evaluator.train_accuracy(truncate_rank(state_dict, mid))
        if acc / baseline >= retention:
            hi = mid
        else:
            lo = mid + 1
    return lo, baseline