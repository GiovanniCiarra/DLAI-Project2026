"""Compression grids and systematic evaluation of a set of weights."""

from grokzip.compression import prune, quantize, truncate_rank

# Rank: refined to the range between 8 and 32, which encompasses both observed thresholds.
# Bits: integers, so resolution is limited by design; the action ranges between 3 and 6.
# Sparsity: step of 0.1 up to 0.7, then sparser, because all the models break down there.
BITS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
SPARSITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
RANKS = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128]

OPERATORS = {
    "quantize": (BITS, quantize),
    "prune": (SPARSITIES, prune),
    "rank": (RANKS, truncate_rank),
}


def sweep_state_dict(state_dict: dict, evaluator, operators=None) -> list[dict]:
    """
    Apply each operator across its entire grid and measure the accuracies.

    Returns a row for each (operator, level) combination, containing the
    training accuracy—the project metric—and the test accuracy, which is
    calculated solely for retrospective validation and never used to construct
    the metric.

    The uncompressed reference is included as a separate row (operator="none"),
    allowing relative retention to be calculated directly from the CSV without
    referring back to the manifest.
    """
    operators = operators or OPERATORS

    train_full, test_full = evaluator.both_accuracies(state_dict)
    rows = [
        {
            "operator": "none",
            "level": float("nan"),
            "train_acc": train_full,
            "test_acc": test_full,
            "train_retention": 1.0,
        }
    ]

    for name, (grid, function) in operators.items():
        for level in grid:
            train_acc, test_acc = evaluator.both_accuracies(function(state_dict, level))
            rows.append(
                {
                    "operator": name,
                    "level": float(level),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_retention": train_acc / train_full if train_full > 0 else 0.0,
                }
            )
    return rows