"""End-to-end check of the scripts on a small synthetic dataset: prepared file -> graphs -> training -> summary."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def make_dataset(path: Path, n=12, days=20, seed=0):
    rng = np.random.default_rng(seed)
    T = days * 288
    road = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        road[i, i + 1] = road[i + 1, i] = 1.0
    daily = 50 + 40 * np.sin(2 * np.pi * np.arange(T) / 288)[:, None]
    flow = daily + rng.standard_normal((T, n)) * 5
    for t in range(3, T):
        flow[t, 1:] += 0.3 * (flow[t - 3, :-1] - daily[t - 3, 0])
    flow = flow.astype(np.float32)
    flow[rng.random((T, n)) < 0.01] = np.nan
    inputs = np.where(np.isfinite(flow), flow, np.nanmean(flow, 0)).astype(np.float32)
    timestamps = 1_609_459_200_000_000_000 + np.arange(T, dtype=np.int64) * 300_000_000_000
    np.savez_compressed(path, flow=flow, inputs=inputs, timestamps=timestamps, sensors=np.arange(n), road=road)


def run(*args):
    proc = subprocess.run([PY, "-W", "ignore", *args], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc.stdout


def test_pipeline_end_to_end(tmp_path):
    data = tmp_path / "toy.npz"
    make_dataset(data)
    out = run("scripts/build_graphs.py", "--data", str(data), "--horizons", "3", "6", "12", "--out", str(tmp_path / "graphs"))
    for h in (3, 6, 12):
        assert (tmp_path / "graphs" / f"G_h{h}.npz").exists() and (tmp_path / "graphs" / f"VAR_h{h}.npz").exists()
    info = json.loads((tmp_path / "graphs" / "G_h3.json").read_text())
    assert info["edges"] == info["var_supported"] + info["added"]

    for config in ("family", "g12_all", "none"):
        for seed in (1, 2):
            run("scripts/train.py", "--config", config, "--seed", str(seed), "--data", str(data), "--graphs", str(tmp_path / "graphs"),
                "--results", str(tmp_path / "results"), "--epochs", "2", "--batch-size", "32")
    results = json.loads((tmp_path / "results" / "family__seed1" / "results.json").read_text())
    assert set(results["test"]) >= {"3", "6", "12", "mean"} and results["best_epoch"] >= 1
    none = json.loads((tmp_path / "results" / "none_nograph__seed1" / "results.json").read_text())
    assert none["parameters"] == results["parameters"] and all(v == 0 for v in none["test_gate"].values())
    for h in (3, 6, 12):
        z = np.load(tmp_path / "results" / "family__seed1" / f"test_predictions_h{h}.npz")
        assert not np.isnan(z["targets"]).any() and z["targets"].shape == z["predictions"].shape

    summary = run("scripts/summarise.py", "--results", str(tmp_path / "results"))
    assert "family" in summary and "paired RMSE ratio family / g12_all" in summary and "seed 1:" in summary


def test_physical_and_semantic_branches(tmp_path):
    """--physical --semantic: extra branches, per-branch gate values, equal parameter counts across configurations."""
    data = tmp_path / "toy.npz"
    make_dataset(data)
    run("scripts/build_graphs.py", "--data", str(data), "--horizons", "3", "6", "12", "--out", str(tmp_path / "graphs"))
    for config in ("family", "none"):
        run("scripts/train.py", "--config", config, "--seed", "1", "--data", str(data), "--graphs", str(tmp_path / "graphs"),
            "--results", str(tmp_path / "results"), "--epochs", "2", "--batch-size", "32", "--physical", "--semantic", "--semantic-top-k", "4")
    fam = json.loads((tmp_path / "results" / "family_phys_sem__seed1" / "results.json").read_text())
    none = json.loads((tmp_path / "results" / "none_nograph_phys_sem__seed1" / "results.json").read_text())
    assert fam["branches"] == ["causal", "physical", "semantic"] and fam["parameters"] == none["parameters"]
    for h in ("3", "6", "12"):
        g = fam["test_gate_branches"][h]
        assert set(g) == {"causal", "physical", "semantic"} and 0 < g["causal"] + g["physical"] + g["semantic"] <= 1
        assert fam["test_gate"][h] == g["causal"] and none["test_gate_branches"][h]["causal"] == 0
    history = (tmp_path / "results" / "family_phys_sem__seed1" / "history.csv").read_text().splitlines()[0]
    assert "gate_physical_h3" in history and "gate_semantic_h12" in history
    plain = json.loads((tmp_path / "results" / "family_phys_sem__seed1" / "config.json").read_text())["train"]["model"]
    assert plain["physical"] and plain["semantic"] and plain["semantic_top_k"] == 4


def test_five_minute_horizon(tmp_path):
    """Adding h=1: graphs, an extra branch and head, per-horizon results, and 'mean' still the mean over 15/30/60 min."""
    data = tmp_path / "toy.npz"
    make_dataset(data)
    run("scripts/build_graphs.py", "--data", str(data), "--horizons", "1", "3", "6", "12", "--out", str(tmp_path / "graphs"))
    assert (tmp_path / "graphs" / "G_h1.npz").exists()
    run("scripts/train.py", "--config", "family", "--seed", "1", "--data", str(data), "--graphs", str(tmp_path / "graphs"),
        "--results", str(tmp_path / "results"), "--epochs", "2", "--batch-size", "32", "--horizons", "1", "3", "6", "12")
    r = json.loads((tmp_path / "results" / "family__seed1" / "results.json").read_text())
    assert set(r["test"]) >= {"1", "3", "6", "12", "mean", "mean_all"}
    for metric in ("mae", "rmse"):
        main = np.mean([r["test"][h][metric] for h in ("3", "6", "12")])
        every = np.mean([r["test"][h][metric] for h in ("1", "3", "6", "12")])
        assert abs(r["test"]["mean"][metric] - main) < 1e-9 and abs(r["test"]["mean_all"][metric] - every) < 1e-9
    assert abs(r["test"]["error_growth_rmse"] - r["test"]["12"]["rmse"] / r["test"]["3"]["rmse"]) < 1e-9
    assert (tmp_path / "results" / "family__seed1" / "test_predictions_h1.npz").exists()
