"""Parallel version of PDFormerDataset._get_dtw: same fastdtw call on the same daily-average curves, so the matrix is identical,
but the rows are spread over all CPU cores. Run once before run_model.py; run_model.py then loads the cache instead of recomputing.

    python precompute_dtw.py --name PEMSBAY [--out path.npy]
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from fastdtw import fastdtw

DATA = None


def load(name):
    df = pd.read_csv(f"./raw_data/{name}/{name}.dyna")[["time", "entity_id", "traffic_speed"]]
    n = df["entity_id"].nunique()
    values = df["traffic_speed"].values.astype(float).reshape(n, -1)
    return values.T[:, :, None]


def init(data_mean):
    global DATA
    DATA = data_mean


def row(i):
    n = DATA.shape[1]
    return i, [fastdtw(DATA[:, i, :], DATA[:, j, :], radius=6)[0] for j in range(i, n)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    out = a.out or f"./libcity/cache/dataset_cache/dtw_{a.name}.npy"
    df = load(a.name)
    day = 24 * (3600 // 300)
    data_mean = np.mean([df[day * i: day * (i + 1)] for i in range(df.shape[0] // day)], axis=0)
    n = data_mean.shape[1]
    dtw = np.zeros((n, n))
    with Pool(a.workers, initializer=init, initargs=(data_mean,)) as pool:
        for k, (i, vals) in enumerate(pool.imap_unordered(row, range(n)), 1):
            dtw[i, i:] = vals
            if k % 25 == 0 or k == n:
                print(f"{k}/{n} rows", flush=True)
    for i in range(n):
        for j in range(i):
            dtw[i][j] = dtw[j][i]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, dtw)
    print("saved", out)
