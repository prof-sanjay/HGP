"""
Part A: horizon-specific causal graphs from the training rows.

    python scripts/build_graphs.py --data data/gba_2021.npz --horizons 3 6 12 --out graphs

For every horizon h this writes
    graphs/G_h{h}.npz      the HGP graph (knockoff-selected edges, VAR / partial-correlation weights)
    graphs/VAR_h{h}.npz    the VAR-only reference graph (top-25 by |Psi_h|)
    graphs/knockoff_h{h}.npz   statistics of every target, for inspection and reproducibility

Control (Part A ablation):  --control pool_var
    Same candidate pool, same VAR / partial-correlation weights, same number of sources per target as the knockoff
    selection, but the sources are the pool members with the largest standardised VAR effect (no knockoff test).
    G_h{h}.npz then holds this control graph and KO_G_h{h}.npz the real knockoff graph, so
    "family with --graphs <this folder>" runs the control with no change to training.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import asdict

from hgp.config import DataConfig, GraphConfig
from hgp.graphs import (assemble, candidate_pool, detrend, fit_var, impulse_responses, knockoff_statistics,
                        partial_correlation, save_graph, select_sources, standardised_effect, var_top_k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=DataConfig.path)
    ap.add_argument("--horizons", type=int, nargs="+", default=[3, 6, 12])
    ap.add_argument("--fdr", type=float, default=GraphConfig.fdr, help="knockoff+ level q")
    ap.add_argument("--group-corr", type=float, default=GraphConfig.group_corr, help="candidates with |r| above this share a group")
    ap.add_argument("--max-sources", type=int, default=GraphConfig.max_sources)
    ap.add_argument("--var-lag", type=int, default=GraphConfig.var_lag)
    ap.add_argument("--var-ridge", type=float, default=GraphConfig.var_ridge)
    ap.add_argument("--out", default=GraphConfig.out_dir)
    ap.add_argument("--control", choices=["pool_var"], default=None,
                    help="pool_var: replace knockoff selection by the top VAR-effect pool members (same count per target)")
    ap.add_argument("--targets", type=int, default=None, help="only the first K targets (quick checks)")
    args = ap.parse_args()
    dc = DataConfig.from_file(args.data)
    gc = GraphConfig(fdr=args.fdr, group_corr=args.group_corr, max_sources=args.max_sources, var_lag=args.var_lag,
                     var_ridge=args.var_ridge, out_dir=args.out)
    out = Path(gc.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(dc.path) as z:
        flow, road, sensors = z["flow"].astype(np.float64), z["road"], z["sensors"]
    train_end = int(len(flow) * dc.train_rate)
    train = flow[:train_end]
    N = flow.shape[1]
    targets = np.arange(N if args.targets is None else min(args.targets, N))
    t0 = time.time()

    coef = fit_var(train, road, gc.var_lag, gc.var_ridge)
    psi = impulse_responses(coef, args.horizons)
    sigma = np.nanstd(train, axis=0)
    sigma[sigma == 0] = 1.0
    pool = candidate_pool(road)
    detrended = detrend(flow, train_end)
    print(f"VAR fitted ({time.time() - t0:.0f}s); pool {pool.sum(1).mean():.1f} candidates per target", flush=True)

    for h in args.horizons:
        t1 = time.time()
        W_grp, W_ind = np.zeros((N, N), np.float32), np.zeros((N, N), np.float32)
        groups, gammas = {}, np.zeros(N, np.float32)
        for k, i in enumerate(targets):
            cand = np.flatnonzero(pool[i])
            if len(cand) == 0:
                continue
            res = knockoff_statistics(detrended, i, cand, h, train_end, gc, np.random.default_rng(1000 + int(i)))
            if res is None:
                continue
            W_grp[i, cand], W_ind[i, cand], groups[i], gammas[i] = res
            if (k + 1) % 100 == 0:
                print(f"  h={h}: {k + 1}/{len(targets)} targets ({time.time() - t1:.0f}s)", flush=True)

        ind_scale = float(np.abs(W_ind).max())
        selected = np.zeros((N, N), dtype=bool)
        for i, grp in groups.items():
            cand = np.flatnonzero(pool[i])
            keep = select_sources(W_grp[i, cand].astype(np.float64), W_ind[i, cand].astype(np.float64), grp, gc.fdr,
                                  gc.max_sources, ind_scale)
            selected[i, cand[keep]] = True

        rho = partial_correlation(train, pool, h, gc.pc_shrink)
        effect = standardised_effect(psi[h], sigma)
        reference = var_top_k(psi[h], gc.max_sources)
        graph, info = assemble(selected, effect, rho, reference)
        if args.control == "pool_var":
            save_graph(out / f"KO_G_h{h}.npz", graph, sensors, h, train_end, fdr=gc.fdr)
            ctrl = np.zeros_like(selected)
            for i in range(N):
                k, cand = int(selected[i].sum()), np.flatnonzero(pool[i])
                if k:
                    ctrl[i, cand[np.lexsort((-rho[i, cand], -effect[i, cand]))[:k]]] = True
            graph, info = assemble(ctrl, effect, rho, reference)
            info["control"] = "pool_var"
            info["overlap_with_knockoff"] = int((ctrl & selected).sum())
        info.update(horizon=h, targets_done=len(groups), seconds=time.time() - t1, graph_config=asdict(gc))
        save_graph(out / f"G_h{h}.npz", graph, sensors, h, train_end, fdr=gc.fdr)
        save_graph(out / f"VAR_h{h}.npz", reference, sensors, h, train_end)
        np.savez_compressed(out / f"knockoff_h{h}.npz", W_group=W_grp, W_individual=W_ind, gamma=gammas, selected=selected,
                            sensors=sensors)
        (out / f"G_h{h}.json").write_text(json.dumps(info, indent=2))
        print(f"h={h:2d}: {info['edges']} edges ({info['var_supported']} VAR-supported, {info['added']} added), "
              f"{info['sources_per_target']:.1f} sources/target, {info['targets_without_sources']} targets without sources "
              f"({info['seconds'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
