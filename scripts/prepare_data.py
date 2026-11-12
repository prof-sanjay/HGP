"""
Prepare a dataset for HGP.

    LargeST-GBA (flow):
        python scripts/prepare_data.py --dataset gba --traffic gba_his_raw_2021.h5 --adjacency gba_rn_adj.npy --out data/gba_2021.npz
    PEMS-BAY (speed):
        python scripts/prepare_data.py --dataset pems_bay --traffic pems-bay.h5 --adjacency adj_mx_bay.pkl --out data/pems_bay.npz

Writes one .npz with
    flow        observed values, NaN where missing
    inputs      the same with short gaps filled; used only as model input, never as a target
    timestamps  nanoseconds since the epoch, 5-minute grid
    sensors     sensor indices
    road        road adjacency (road[i, j] != 0: a road link between i and j)
    dataset     name of the dataset; selects its split and metric settings (hgp/config.py)

PEMS-BAY: speed in mph, a reading of 0 means missing (DCRNN convention), adjacency is the pickled
[sensor_ids, id_to_index, adj_mx] of DCRNN. Gaps of at most six steps (30 minutes) are linearly interpolated inside
each split for the inputs; longer gaps stay missing.

GBA with --implementation pointing at the thesis code base uses that pipeline's sensor filter and gap-band filling, so
the data matches the thesis experiments exactly; without it all sensors are kept and the same short-gap
interpolation is used.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgp.config import DATASETS
from hgp.graphs.pool import candidate_pool


def interpolate_short_gaps(x: np.ndarray, max_gap: int = 6) -> np.ndarray:
    out = x.copy()
    idx = np.arange(len(x))
    for j in range(x.shape[1]):
        col = out[:, j]
        missing = ~np.isfinite(col)
        if not missing.any() or missing.all():
            continue
        filled = np.interp(idx, idx[~missing], col[~missing])
        edges = np.diff(np.concatenate([[0], missing.astype(int), [0]]))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if end - start <= max_gap and start > 0 and end < len(col):
                col[start:end] = filled[start:end]
    return out


def read_gba(traffic: str, adjacency: str):
    import h5py
    with h5py.File(traffic, "r") as f:
        flow = np.asarray(f["traffic"][:], dtype=np.float32)
        timestamps = np.asarray(f["timestamps"][:], dtype=np.int64)
    return flow, timestamps, np.load(adjacency).astype(np.float32)


def read_pems_bay(traffic: str, adjacency: str):
    import pandas as pd
    table = pd.read_hdf(traffic)
    flow = table.values.astype(np.float32)
    flow[flow == 0] = np.nan
    timestamps = table.index.values.astype("datetime64[ns]").astype(np.int64)
    with open(adjacency, "rb") as f:
        content = pickle.load(f, encoding="latin1")
    road = np.asarray(content[2] if isinstance(content, (list, tuple)) else content, dtype=np.float32)
    return flow, timestamps, road


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--traffic", required=True)
    ap.add_argument("--adjacency", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--implementation", default=None, help="GBA only: thesis code base for the identical sensor filter and filling")
    ap.add_argument("--subset", type=int, default=None, help="keep only K sensors (quick checks)")
    args = ap.parse_args()
    if args.implementation and args.dataset != "gba":
        raise SystemExit("--implementation applies to the gba dataset only")

    flow, timestamps, road = (read_gba if args.dataset == "gba" else read_pems_bay)(args.traffic, args.adjacency)
    if road.shape != (flow.shape[1],) * 2:
        raise SystemExit(f"adjacency {road.shape} does not match {flow.shape[1]} sensors")
    preset = DATASETS[args.dataset]
    T = len(flow)
    train_end = int(T * preset["train_rate"])
    val_end = train_end + int(T * preset["val_rate"])
    bounds = [(0, train_end), (train_end, val_end), (val_end, T)]

    if args.implementation:
        sys.path.insert(0, args.implementation)
        from data.imputation import fill_split, fit_neighbour_models
        from data.sensor_filter import cached_duplicate_filter
        sensors, _ = cached_duplicate_filter(flow[:train_end], Path(args.implementation) / "outputs" / "sensor_filter")
    else:
        sensors = np.arange(flow.shape[1])
    if args.subset:
        sensors = np.sort(np.random.default_rng(0).choice(sensors, min(args.subset, len(sensors)), replace=False))
    flow, road = flow[:, sensors], road[np.ix_(sensors, sensors)]

    inputs = np.empty_like(flow)
    if args.implementation:
        models = fit_neighbour_models(flow[:train_end], road, num_neighbours=5)
        for (lo, hi), look_ahead in zip(bounds, (True, False, False)):
            inputs[lo:hi] = fill_split(flow[lo:hi], models, look_ahead)[0]
    else:
        for lo, hi in bounds:
            inputs[lo:hi] = interpolate_short_gaps(flow[lo:hi])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, flow=flow, inputs=inputs, timestamps=timestamps, sensors=np.asarray(sensors, np.int64),
                        road=road, dataset=args.dataset)

    links = ((road != 0) | (road != 0).T)
    np.fill_diagonal(links, False)
    pool = candidate_pool(road)
    print(f"{args.out}: {args.dataset}, {T} steps x {flow.shape[1]} sensors; missing {np.isnan(flow).mean():.4%} of values, "
          f"{np.isnan(inputs).mean():.4%} after filling")
    print(f"split: train {train_end}, validation {val_end - train_end}, test {T - val_end} steps")
    print(f"road links per sensor {links.sum(1).mean():.1f}; candidate pool per sensor {pool.sum(1).mean():.1f} of {flow.shape[1] - 1}")
    if pool.sum(1).mean() > 0.5 * (flow.shape[1] - 1):
        print("warning: the pool covers most of the network, so knockoff selection has little structure to exploit")


if __name__ == "__main__":
    main()
