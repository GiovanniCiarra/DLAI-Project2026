"""
Modular addition dataset.

The task is to calculate (a + b) mod p. Each example is a sequence of three tokens:
[a, b, EQUALS], where EQUALS is a dedicated token with index p. The model reads the
final position and produces a logit for each of the p possible classes.
"""

import torch

EQUALS_OFFSET = 0


def equals_token(p: int) -> int:
    """Index of the '=' token. It equals p — that is, the first free index after the digits."""
    return p


def vocab_size(p: int) -> int:
    """Vocabulary size: the p digits plus the '=' token."""
    return p + 1


def make_dataset(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs all p^2 pairs (a, b) and their respective targets.

    Returns:
        inputs:  (p^2, 3) tensor of token indices
        targets: (p^2,) tensor containing (a + b) mod p

    The order is deterministic: a varies slowly, b quickly. Shuffling is not
    performed here, as that is the responsibility of split_dataset, which uses
    its own seed.
    """
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, equals_token(p))

    inputs = torch.stack([a, b, eq], dim=1)
    targets = (a + b) % p
    return inputs, targets


def split_dataset(
    n_examples: int, train_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Splits the indices into training and test sets deterministically.

    Uses a CPU-based generator with an explicit seed: the same seed produces the same
    split on any machine and independently of the global torch state.
    """
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_examples, generator=generator)
    n_train = int(n_examples * train_fraction)
    return perm[:n_train], perm[n_train:]