"""
Execution of experimental conditions with checkpoint saving.

Conditions:
  main   fraction 0.30, weight decay 1.0, 8 seed   -> main experiment
  long   fraction 0.25, weight decay 1.0, 4 seed   -> long plateau, sturdiness
  nc_wd0 fraction 0.30, weight decay 0.0, 2 seed   -> control: is it just the long training?
  nc_mem fraction 0.20, weight decay 1.0, 2 seed   -> control: is it just weight decay?

"""

import argparse

from grokzip.checkpoints import checkpoint_steps
from grokzip.train import (
    Config,
    gate1_passed,
    grokking_summary,
    pick_device,
    save_history,
    save_manifest,
    train_run,
)

CONDITIONS = {
    "main": dict(
        train_fraction=0.30, weight_decay=1.0, steps=30_000, seeds=range(8),
        dense_from=2_000, dense_to=14_000, n_dense=80,
    ),
    "long": dict(
        train_fraction=0.25, weight_decay=1.0, steps=40_000, seeds=range(4),
        dense_from=8_000, dense_to=30_000, n_dense=146,
    ),
    "nc_wd0": dict(
        train_fraction=0.30, weight_decay=0.0, steps=30_000, seeds=range(2),
        dense_from=2_000, dense_to=14_000, n_dense=80,
    ),
    "nc_mem": dict(
        train_fraction=0.20, weight_decay=1.0, steps=30_000, seeds=range(2),
        dense_from=2_000, dense_to=14_000, n_dense=80,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=None, help="sovrascrive i seed di default"
    )
    args = parser.parse_args()

    spec = CONDITIONS[args.condition]
    seeds = args.seeds if args.seeds is not None else list(spec["seeds"])
    device = pick_device()
    print(f"device: {device}")
    print(f"condizione: {args.condition} | seed: {seeds}\n")

    steps_grid = checkpoint_steps(
        spec["steps"],
        dense_from=spec["dense_from"],
        dense_to=spec["dense_to"],
        n_dense=spec["n_dense"],
    )
    print(f"checkpoint per run: {len(steps_grid)}\n")

    results = []
    for seed in seeds:
        run_name = f"{args.condition}_seed{seed}"
        print(f"===== {run_name} =====")
        config = Config(
            seed=seed,
            steps=spec["steps"],
            train_fraction=spec["train_fraction"],
            weight_decay=spec["weight_decay"],
        )

        history, manifest = train_run(
            config,
            device,
            verbose=False,
            checkpoint_dir=f"checkpoints/{run_name}",
            checkpoint_at=steps_grid,
        )

        save_history(history, config, f"results/{run_name}_history.csv")
        save_manifest(manifest, f"results/{run_name}_manifest.csv")

        summary = grokking_summary(history)
        summary["run"] = run_name
        summary["gate"] = gate1_passed(summary)
        results.append(summary)

        def fmt(v):
            return "mai" if v is None else str(v)

        print(
            f"  t_mem {fmt(summary['t_mem'])} | t_grok50 {fmt(summary['t_grok50'])} "
            f"| altopiano {fmt(summary['plateau'])} | test max "
            f"{summary['max_test_acc']:.4f} | gate {'VERDE' if summary['gate'] else 'rosso'}\n"
        )

    print("=" * 76)
    print(f"{'run':>18} {'t_mem':>7} {'t_grok50':>9} {'altopiano':>10} {'gate':>7}")
    print("-" * 76)
    for row in results:
        def fmt(v, w):
            return f"{'mai':>{w}}" if v is None else f"{v:>{w}}"

        print(
            f"{row['run']:>18} {fmt(row['t_mem'], 7)} {fmt(row['t_grok50'], 9)} "
            f"{fmt(row['plateau'], 10)} {'VERDE' if row['gate'] else 'rosso':>7}"
        )
    print("=" * 76)

    if args.condition == "main":
        n_ok = sum(r["gate"] for r in results)
        print(f"\nGATE 2: {n_ok}/{len(results)} seed passano (richiesti almeno 6)")
        print(f"GATE 2: {'VERDE' if n_ok >= 6 else 'ROSSO'}")
    elif args.condition.startswith("nc_"):
        max_test = max(r["max_test_acc"] for r in results)
        print(f"\nControllo negativo, test accuracy massima: {max_test:.4f}")
        print(f"GATE controllo: {'VERDE' if max_test < 0.20 else 'ROSSO'} (atteso < 0.20)")


if __name__ == "__main__":
    main()