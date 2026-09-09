"""

Automatic selection of anchor checkpoints.
"""

PLATEAU_TRAIN_MIN = 0.99
PLATEAU_TEST_MAX = 0.15


def pick_anchors(manifest: list[dict]) -> dict[str, dict]:
    """
    Returns the available anchors for a run.

    early : first plateau checkpoint (train >= 0.99 and test <= 0.15)
    late  : last plateau checkpoint
    final : last checkpoint of the run

    For runs that never generalize, `late` and `final` coincide, and only
    the `late` entry is returned.
    """
    plateau = [
        row
        for row in manifest
        if row["train_acc"] >= PLATEAU_TRAIN_MIN and row["test_acc"] <= PLATEAU_TEST_MAX
    ]
    if not plateau:
        raise RuntimeError("nessun checkpoint in fase di memorizzazione")

    anchors = {"early": plateau[0], "late": plateau[-1]}
    final = manifest[-1]
    if final["step"] != anchors["late"]["step"]:
        anchors["final"] = final
    return anchors


def generalises(anchor: dict) -> bool:
    return anchor["test_acc"] >= 0.90