"""
A single-layer Transformer for modular arithmetic.

It follows the simplified architecture used in the literature to study grokking:
a single block, no LayerNorm, no dropout, and ReLU activation. The absence of
LayerNorm is not merely a convenient simplification; it allows the embedding
matrix to be directly analyzed in the Fourier basis—an analysis that per-token
normalization would render ambiguous.
"""

import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Causal multi-head self-attention, without bias and without dropout."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model deve essere divisibile per n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, d_model = x.shape

        def split_heads(t):
            return t.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(self.W_Q(x))
        k = split_heads(self.W_K(x))
        v = split_heads(self.W_V(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        causal = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        weights = scores.softmax(dim=-1)

        out = weights @ v
        out = out.transpose(1, 2).reshape(batch, seq, d_model)
        return self.W_O(out)


class MLP(nn.Module):
    """Two-layer feed-forward network with ReLU, without bias."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp, bias=False)
        self.W_out = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_out(torch.relu(self.W_in(x)))


class GrokkingTransformer(nn.Module):
    """Single-block Transformer. Reads the final position and classifies into n_classes."""

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        seq_len: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        d_mlp: int = 512,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.empty(seq_len, d_model))
        self.attention = SelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_mlp)
        self.unembed = nn.Linear(d_model, n_classes, bias=False)

        self._init_weights(d_model)

    def _init_weights(self, d_model: int) -> None:
        """
        Gaussian initialization with a standard deviation of 1/sqrt(fan_in).

        This is the scale used in the reference implementation. A different scale
        can shift the grokking time by as much as an order of magnitude, so
        there is no room for improvisation here.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=1.0 / math.sqrt(module.in_features))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.pos_embed, std=1.0 / math.sqrt(d_model))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) + self.pos_embed
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return self.unembed(x[:, -1, :])