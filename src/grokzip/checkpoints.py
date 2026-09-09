"""
Checkpoint grid and weight saving along the trajectory.

The grid is hybrid:
  - 120 points on a logarithmic scale across the entire training run, to cover the
    initial decades (where memorization occurs) with uniform density without
    wasting points in the tail;
  - 80 points on a linear scale within a fixed window around the transition, to
    precisely resolve the time lag between the drop in the compression threshold
    and the jump in test accuracy.

The dense window is identical for all conditions, including negative controls:
providing higher resolution for runs that "grok" would skew the comparison.
"""

from pathlib import Path

import numpy as np
import torch


def checkpoint_steps(
    total_steps: int,
    n_log: int = 120,
    dense_from: int | None = None,
    dense_to: int | None = None,
    n_dense: int = 80,
) -> list[int]:
    """
    Constructs the grid of steps at which to save the weights.

    Always includes step 0 and the final step. Duplicates between the two grids
    are merged.
    """
    steps = {0, total_steps}
    steps.update(int(round(v)) for v in np.geomspace(1, total_steps, n_log))

    if dense_from is not None and dense_to is not None:
        if dense_from >= dense_to:
            raise ValueError("dense_from deve essere minore di dense_to")
        steps.update(int(round(v)) for v in np.linspace(dense_from, dense_to, n_dense))

    return sorted(s for s in steps if 0 <= s <= total_steps)


def checkpoint_path(directory: str | Path, step: int) -> Path:
    return Path(directory) / f"step_{step:07d}.pt"


def save_checkpoint(model, directory: str | Path, step: int) -> Path:
    """Save the weights to the CPU in float32."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone().float() for k, v in model.state_dict().items()}
    path = checkpoint_path(directory, step)
    torch.save(state, path)
    return path


def load_checkpoint(directory: str | Path, step: int) -> dict:
    """Reload a checkpoint as a dictionary of tensors on the CPU."""
    return torch.load(checkpoint_path(directory, step), map_location="cpu")