"""
Evaluation of a compressed state_dict on run data.

The model is built once, and the weights are reloaded for each
measurement.
"""

import csv
from pathlib import Path

import torch

from grokzip.data import make_dataset, split_dataset, vocab_size
from grokzip.model import GrokkingTransformer
from grokzip.train import Config


def load_config(history_path: str | Path) -> Config:
    """
    Reconstructs the configuration from the first line of a trajectory CSV file.

    This prevents run parameters from being manually rewritten during analysis,
    which would risk a mismatch between the trained model and the one
    being evaluated.
    """
    with open(history_path, newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    fields = Config.__dataclass_fields__
    values = {}
    for name, field in fields.items():
        raw = row[name]
        values[name] = int(raw) if field.type in ("int", int) else float(raw)
    return Config(**values)


def read_manifest(path: str | Path) -> list[dict]:
    """Reads the checkpoint manifest, converting the numeric fields."""
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "step": int(row["step"]),
                    "path": row["path"],
                    "train_acc": float(row["train_acc"]),
                    "test_acc": float(row["test_acc"]),
                    "train_loss": float(row["train_loss"]),
                    "test_loss": float(row["test_loss"]),
                    "param_norm": float(row["param_norm"]),
                }
            )
    return rows


class Evaluator:
    """
    Measures the accuracy of a state_dict on training and test data.

    Training accuracy is the project's core metric: the compression threshold
    is defined based on it and therefore does not require a test set.
    Test accuracy is calculated solely to validate the detector post-hoc,
    never to construct it.
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device

        inputs, targets = make_dataset(config.p)
        train_idx, test_idx = split_dataset(
            len(inputs), config.train_fraction, seed=config.seed
        )
        self.train_inputs = inputs[train_idx].to(device)
        self.train_targets = targets[train_idx].to(device)
        self.test_inputs = inputs[test_idx].to(device)
        self.test_targets = targets[test_idx].to(device)

        self.model = GrokkingTransformer(
            vocab_size=vocab_size(config.p),
            n_classes=config.p,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_mlp=config.d_mlp,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def _accuracy(self, inputs, targets) -> float:
        logits = self.model(inputs)
        return (logits.argmax(dim=-1) == targets).float().mean().item()

    def train_accuracy(self, state_dict: dict) -> float:
        """Accuracy on training data only. No test set involved."""
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()}
        )
        return self._accuracy(self.train_inputs, self.train_targets)

    def both_accuracies(self, state_dict: dict) -> tuple[float, float]:
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()}
        )
        return (
            self._accuracy(self.train_inputs, self.train_targets),
            self._accuracy(self.test_inputs, self.test_targets),
        )