"""
Summarise finished runs: per configuration and horizon, mean and standard deviation over seeds.

    python scripts/summarise.py --results results [--metric rmse] [--csv table.csv]

Also prints the paired comparison that decides the main claim (family against g12_all): the RMSE ratio per
horizon with a 95% bootstrap interval over sensors, computed within each seed from the saved test predictions.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_runs(results: Path) -> dict:
    runs = defaultdict(dict)
    for path in sorted(results.glob("*/results.json")):
        r = json.loads(path.read_text())
        name, seed = path.parent.name.rsplit("__seed", 1)
        runs[name][int(seed)] = r
    return runs


def table(runs: dict, metric: str) -> list[dict]:
    rows = []
    for name, by_seed in runs.items():
        horizons = [h for h in next(iter(by_seed.values()))["test"] if h.isdigit()]
        row = {"configuration": name, "seeds": len(by_seed), "parameters": next(iter(by_seed.values()))["parameters"]}
        for h in horizons + ["mean"]:
            values = np.array([r["test"][h][metric] for r in by_seed.values()])
            row[f"h{h}" if h != "mean" else "mean"] = f"{values.mean():.3f} ± {values.std(ddof=1):.3f}" if len(values) > 1 else f"{values[0]:.3f}"
        row["gate"] = " / ".join(f"{np.mean([r['test_gate'][h] for r in by_seed.values()]):.2f}" for h in horizons)
        rows.append(row)
    return rows


def paired_ratio(results: Path, a: str, b: str, seed: int, horizon: int, boots: int = 2000):
    """RMSE(a) / RMSE(b) on identical test windows, with a bootstrap interval over sensors."""
    za = np.load(results / f"{a}__seed{seed}" / f"test_predictions_h{horizon}.npz")
    zb = np.load(results / f"{b}__seed{seed}" / f"test_predictions_h{horizon}.npz")
    if not np.array_equal(za["rows"], zb["rows"]):
        raise ValueError("runs were scored on different windows")
    se_a = ((za["predictions"] - za["targets"]) ** 2).mean(0)
    se_b = ((zb["predictions"] - zb["targets"]) ** 2).mean(0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(se_a), (boots, len(se_a)))
    ratios = np.sqrt(se_a[idx].mean(1) / se_b[idx].mean(1))
    return float(np.sqrt(se_a.mean() / se_b.mean())), tuple(np.percentile(ratios, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--metric", default="rmse", choices=["rmse", "mae", "mape"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--compare", nargs=2, default=["family", "g12_all"], metavar=("A", "B"))
    args = ap.parse_args()
    results = Path(args.results)
    runs = load_runs(results)
    if not runs:
        raise SystemExit(f"no runs under {results}")

    rows = table(runs, args.metric)
    keys = list(rows[0])
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    print(f"test {args.metric.upper()} (mean ± sd over seeds); gate = mean gate value per horizon")
    print("  ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in keys))
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    a, b = args.compare
    if a in runs and b in runs:
        print(f"\npaired RMSE ratio {a} / {b} (95% bootstrap over sensors; below 1 favours {a})")
        for seed in sorted(set(runs[a]) & set(runs[b])):
            horizons = [int(h) for h in runs[a][seed]["test"] if h.isdigit()]
            parts = []
            for h in horizons:
                ratio, (lo, hi) = paired_ratio(results, a, b, seed, h)
                parts.append(f"h{h}: {ratio:.4f} [{lo:.4f}, {hi:.4f}]")
            print(f"  seed {seed}: " + "   ".join(parts))


if __name__ == "__main__":
    main()
