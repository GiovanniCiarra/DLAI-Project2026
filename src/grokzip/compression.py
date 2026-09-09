"""
Compression operators applied post-hoc to a set of weights.

All three operators:
  - receive and return a name-to-tensor dictionary;
  - do not modify the input;
  - preserve keys and shapes exactly.
"""

import torch


def _check_state_dict(state_dict: dict) -> None:
    if not state_dict:
        raise ValueError("state_dict vuoto")


def quantize(state_dict: dict, bits: int) -> dict:
    """
    Symmetric per-tensor uniform quantization with absmax scaling.

    For each tensor, the scale is max|W| / (2^(bits-1) - 1), and weights are
    rounded to the nearest level. There are thus 2^bits - 1 representable
    levels, symmetric around zero (zero is always representable).

    A key property is that the operator is scale-invariant—meaning
    quantize(c*W, b) = c*quantize(W, b) for any c != 0. This implies by
    design that a uniform reduction in weight magnitude cannot, on its own,
    improve robustness to quantization.

    bits >= 2: at 1 bit, the only representable level would be zero.
    """
    _check_state_dict(state_dict)
    if bits < 2:
        raise ValueError("bits deve essere almeno 2")

    n_levels = 2 ** (bits - 1) - 1
    output = {}
    for name, weight in state_dict.items():
        absmax = weight.abs().max()
        if absmax == 0:
            output[name] = weight.clone()
            continue
        scale = absmax / n_levels
        levels = torch.clamp(torch.round(weight / scale), -n_levels, n_levels)
        output[name] = levels * scale
    return output


def _global_keep_mask(state_dict: dict, sparsity: float) -> dict:
    """
    Boolean mask of weights to be retained, using a single threshold across the entire network.

    Uses an explicit ordering instead of a numerical threshold: with a threshold,
    weights with exactly the same magnitude would all be zeroed out together,
    and the effective sparsity would exceed the target level.   
    """
    flat = torch.cat([w.abs().flatten() for w in state_dict.values()])
    n_prune = int(round(sparsity * flat.numel()))

    keep_flat = torch.ones(flat.numel(), dtype=torch.bool)
    if n_prune > 0:
        keep_flat[torch.argsort(flat)[:n_prune]] = False

    masks, offset = {}, 0
    for name, weight in state_dict.items():
        size = weight.numel()
        masks[name] = keep_flat[offset : offset + size].view_as(weight)
        offset += size
    return masks


def prune(state_dict: dict, sparsity: float, scope: str = "global") -> dict:
    """
    One-shot magnitude pruning: zeroes out the fraction of smallest weights specified by `sparsity`.

    scope="global"     a single threshold for the entire network (standard operator in
                       pruning literature);
    scope="per_tensor" uniform sparsity within each tensor; used as a control
                       because the 1/sqrt(fan_in) initialization gives tensors
                       different scales, meaning global pruning would affect
                       some before others.

    No retraining after pruning: the operator measures weight fragility
    in their current state.
    """
    _check_state_dict(state_dict)
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity deve stare in [0, 1]")
    if scope not in ("global", "per_tensor"):
        raise ValueError("scope deve essere 'global' o 'per_tensor'")

    if sparsity == 0.0:
        return {name: w.clone() for name, w in state_dict.items()}

    if scope == "global":
        masks = _global_keep_mask(state_dict, sparsity)
    else:
        masks = {
            name: _global_keep_mask({name: w}, sparsity)[name]
            for name, w in state_dict.items()
        }

    return {name: w * masks[name] for name, w in state_dict.items()}


def truncate_rank(state_dict: dict, rank: int) -> dict:
    """
    SVD truncation for matrices: retains the first `rank` singular values.

    Tensors with a maximum rank already lower than `rank` remain unchanged. In
    the model used, all parameters are two-dimensional matrices; the positional
    embedding has the shape (3, 128) and thus a rank of at most 3, meaning that
    truncation to a rank >= 3 acts as the identity operation for it.
    """
    _check_state_dict(state_dict)
    if rank < 1:
        raise ValueError("rank deve essere almeno 1")

    output = {}
    for name, weight in state_dict.items():
        if weight.dim() != 2 or rank >= min(weight.shape):
            output[name] = weight.clone()
            continue
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        output[name] = (u[:, :rank] * s[:rank]) @ vh[:rank]
    return output


def n_parameters(state_dict: dict) -> int:
    return sum(w.numel() for w in state_dict.values())


def actual_sparsity(state_dict: dict) -> float:
    """Fraction of exactly zero weights. It is used to verify the operators."""
    zeros = sum(int((w == 0).sum()) for w in state_dict.values())
    return zeros / n_parameters(state_dict)