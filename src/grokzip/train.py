"""Full-batch training cycle and trajectory recording."""

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from grokzip.data import make_dataset, split_dataset, vocab_size
from grokzip.model import GrokkingTransformer
from grokzip.checkpoints import save_checkpoint


@dataclass
class Config:
    """All parameters of a run. It is serialized into the CSV for traceability."""

    seed: int = 0
    p: int = 97
    train_fraction: float = 0.4
    steps: int = 40_000
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    log_every: int = 25


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, inputs, targets) -> tuple[float, float]:
    """Loss and accuracy on a given set, without gradients."""
    was_training = model.training
    model.eval()
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets).item()
    accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
    model.train(was_training)
    return loss, accuracy


@torch.no_grad()
def parameter_norm(model) -> float:
    """
    L2 norm of all parameters.
    """
    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        total += param.pow(2).sum()
    return total.sqrt().item()


def train_run(
    config: Config,
    device: torch.device,
    verbose: bool = True,
    checkpoint_dir: str | Path | None = None,
    checkpoint_at: list[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Executes a full run.

    Returns two lists:
      history  - dense trajectory (every config.log_every step), to pinpoint
                 memorization and grokking times;
      manifest - one line per saved checkpoint, with accuracies measured
                 exactly at that step.
    """
    torch.manual_seed(config.seed)

    inputs, targets = make_dataset(config.p)
    train_idx, test_idx = split_dataset(
        len(inputs), config.train_fraction, seed=config.seed
    )

    train_inputs = inputs[train_idx].to(device)
    train_targets = targets[train_idx].to(device)
    test_inputs = inputs[test_idx].to(device)
    test_targets = targets[test_idx].to(device)

    model = GrokkingTransformer(
        vocab_size=vocab_size(config.p),
        n_classes=config.p,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_mlp=config.d_mlp,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )

    save_set = set(checkpoint_at) if checkpoint_at else set()
    if save_set and checkpoint_dir is None:
        raise ValueError("checkpoint_at richiede checkpoint_dir")

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"parametri: {n_params:,}")
        print(f"training: {len(train_idx)} esempi | test: {len(test_idx)} esempi")
        if save_set:
            print(f"checkpoint previsti: {len(save_set)}")

    history: list[dict] = []
    manifest: list[dict] = []
    start = time.time()

    for step in range(config.steps + 1):
        logits = model(train_inputs)
        loss = F.cross_entropy(logits, train_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % config.log_every == 0:
            with torch.no_grad():
                train_acc = (
                    (logits.argmax(dim=-1) == train_targets).float().mean().item()
                )
            test_loss, test_acc = evaluate(model, test_inputs, test_targets)
            history.append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "param_norm": parameter_norm(model),
                }
            )

        if step in save_set:

            ck_train_loss, ck_train_acc = evaluate(model, train_inputs, train_targets)
            ck_test_loss, ck_test_acc = evaluate(model, test_inputs, test_targets)
            path = save_checkpoint(model, checkpoint_dir, step)
            manifest.append(
                {
                    "step": step,
                    "path": str(path).replace("\\", "/"),
                    "train_loss": ck_train_loss,
                    "train_acc": ck_train_acc,
                    "test_loss": ck_test_loss,
                    "test_acc": ck_test_acc,
                    "param_norm": parameter_norm(model),
                }
            )

        if verbose and step == 200:
            elapsed = time.time() - start
            print(
                f"ritmo su 200 step -> durata stimata "
                f"{elapsed / 200 * config.steps / 60:.1f} minuti"
            )

        if verbose and step > 0 and step % 5000 == 0:
            last = history[-1]
            print(
                f"step {step:>6} | train {last['train_acc']:.4f} "
                f"| test {last['test_acc']:.4f} | norma {last['param_norm']:.1f}"
            )

    if verbose:
        print(f"completata in {(time.time() - start) / 60:.1f} minuti")
    return history, manifest


def first_step_at_least(history: list[dict], key: str, threshold: float) -> int | None:
    """First step at which `key` reaches the threshold; None if this never happens."""
    for row in history:
        if row[key] >= threshold:
            return row["step"]
    return None


def save_history(history: list[dict], config: Config, path: str | Path) -> None:
    """Writes the history to CSV, with the configuration replicated on each line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config_fields = asdict(config)
    fieldnames = list(history[0].keys()) + list(config_fields.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({**row, **config_fields})

def grokking_summary(history: list[dict]) -> dict:
    """
    Summarizes the trajectory in terms of the numbers defining the regime.

    t_mem      : first step where train accuracy >= 0.99
    t_grok50   : first step where test accuracy  >= 0.50
    t_grok95   : first step where test accuracy  >= 0.95
    plateau    : t_grok50 - t_mem, i.e., the duration of the phase during which the network
                 memorizes without generalizing. This is the window where
                 compressibility measures will be evaluated, so it must be wide.
    delay_ratio: t_grok95 / t_mem, a dimensionless measure of the delay.

    Fields are set to None when the threshold is never reached.
    """
    t_mem = first_step_at_least(history, "train_acc", 0.99)
    t_grok50 = first_step_at_least(history, "test_acc", 0.50)
    t_grok95 = first_step_at_least(history, "test_acc", 0.95)

    plateau = None
    if t_mem is not None and t_grok50 is not None:
        plateau = t_grok50 - t_mem

    delay_ratio = None
    if t_mem is not None and t_grok95 is not None and t_mem > 0:
        delay_ratio = t_grok95 / t_mem

    return {
        "t_mem": t_mem,
        "t_grok50": t_grok50,
        "t_grok95": t_grok95,
        "plateau": plateau,
        "delay_ratio": delay_ratio,
        "max_test_acc": max(row["test_acc"] for row in history),
        "final_test_acc": history[-1]["test_acc"],
    }


def gate1_passed(summary: dict) -> bool:
    """
    The original criterion merely required that grokking occur. The pilot run
    showed that the real risk is not the absence of grokking but its compression:
    the network can generalize almost immediately, leaving a plateau too short
    to sample its compressibility. The gate now explicitly requires a plateau.
    """
    return (
        summary["t_mem"] is not None
        and summary["t_mem"] <= 1_000
        and summary["t_grok50"] is not None
        and summary["t_grok50"] >= 5_000
        and summary["t_grok95"] is not None
        and summary["t_grok95"] <= 30_000
        and summary["plateau"] is not None
        and summary["plateau"] >= 4_000
    )

def save_manifest(manifest: list[dict], path: str | Path) -> None:
    """Writes the checkpoint manifest in CSV format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)