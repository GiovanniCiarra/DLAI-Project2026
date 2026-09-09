"""
Fourier analysis of the embedding matrix.

The circuit known for modular addition represents each number $a$ as a
combination of sines and cosines of $2\pi ka/p$ for a few frequencies $k$. If the model has
constructed that circuit, the embedding energy concentrates on a few frequencies.

The analysis is possible because the model does not contain LayerNorm: per-token
normalization would make the interpretation of the spectrum ambiguous.
"""

import torch

ENERGY_RATIO = 10.0  # threshold declared a priori: energy > 10x the median

def digit_embeddings(state_dict: dict, p: int) -> torch.Tensor:
    """
    The first p rows of the embedding matrix, one for each number from 0 to p-1.

    The last row corresponds to the '=' token and is excluded: it does not represent a number
    and does not participate in the cyclic structure.
    """
    return state_dict["embed.weight"][:p].float()


def frequency_energy(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Energy per frequency, summed across all model dimensions.

    The DFT is applied along the feature axis. The DC component (frequency 0)
    is excluded: it represents the embedding mean, not a cyclical structure.

    Returns a vector of length (p-1)//2: for a real-valued signal, the spectrum is
    symmetric, so k and p-k represent the same frequency and are counted only once.
    """
    p = embeddings.shape[0]
    spectrum = torch.fft.fft(embeddings, dim=0)
    energy = spectrum.abs().pow(2).sum(dim=1)
    return energy[1 : (p + 1) // 2]


def key_frequencies(energy: torch.Tensor, ratio: float = ENERGY_RATIO) -> list[int]:
    """
    Indices of frequencies whose energy exceeds the median by a factor of `ratio`.

    The median serves as a robust reference for the background noise: with only a few
    dominant frequencies amidst dozens of weak ones, it is not skewed upwards.

    The indices are 1-based, meaning they correspond to the circuit's k-th frequency.
    """
    threshold = ratio * energy.median()
    return [int(i) + 1 for i in torch.nonzero(energy > threshold).flatten()]


def concentration(energy: torch.Tensor, top_k: int = 6) -> float:
    """
    Fraction of energy contained in the top-k strongest frequencies.

    A continuous measure of Fourier structure, complementary to the discrete count
    of key frequencies: it does not depend on any threshold.
    """
    total = energy.sum()
    if total == 0:
        return 0.0
    return float(energy.sort(descending=True).values[:top_k].sum() / total)