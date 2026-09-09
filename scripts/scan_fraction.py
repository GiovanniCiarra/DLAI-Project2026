"""
A scan over the training data fraction.

This scan identifies the fraction that places the setup in the 
grokking regime, while simultaneously producing the project's phase diagram.
"""

import argparse

from grokzip.train import (
    Config,
    gate1_passed,
    grokking_summary,
    pick_device,
    save_history,
    train_run,
)


def classify(summary: dict) -> str:
    """
Label the observed regime, following Liu's phase diagram. """
    if summary["t_mem"] is None:
        return "confusione (non memorizza)"
    if summary["t_grok50"] is None:
        return "memorizzazione (non generalizza)"
    if summary["plateau"] is not None and summary["plateau"] < 1_000:
        return "comprensione (nessun altopiano)"
    return "GROKKING"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.20, 0.25, 0.30, 0.35]
    )
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}\n")

    rows = []
    for fraction in args.fractions:
        print(f"===== frazione {fraction:.2f} =====")
        config = Config(seed=args.seed, steps=args.steps, train_fraction=fraction)
        history, _ = train_run(config, device, verbose=False)
        path = f"results/scan_frac{int(fraction * 100):02d}_seed{args.seed}.csv"
        save_history(history, config, path)

        summary = grokking_summary(history)
        summary["fraction"] = fraction
        summary["regime"] = classify(summary)
        summary["gate1"] = gate1_passed(summary)
        rows.append(summary)
        print(f"  {summary['regime']}  ->  {path}\n")

    def fmt(value, width):
        return f"{'mai':>{width}}" if value is None else f"{value:>{width}}"

    print("\n" + "=" * 88)
    print(
        f"{'frac':>6} {'t_mem':>8} {'t_grok50':>10} {'t_grok95':>10} "
        f"{'plateau':>9} {'gate1':>7}  regime"
    )
    print("-" * 88)
    for row in rows:
        print(
            f"{row['fraction']:>6.2f} {fmt(row['t_mem'], 8)} {fmt(row['t_grok50'], 10)} "
            f"{fmt(row['t_grok95'], 10)} {fmt(row['plateau'], 9)} "
            f"{'VERDE' if row['gate1'] else 'rosso':>7}  {row['regime']}"
        )
    print("=" * 88)

    verdi = [r["fraction"] for r in rows if r["gate1"]]
    if verdi:
        print(f"\nFrazioni che passano: {verdi}")
    else:
        print("\nNessuna frazione passa.")


if __name__ == "__main__":
    main()