"""PEMS-BAY -> the input files DSTFGCN's train.py reads. Data preparation only; no model code is touched.

    python prepare_pemsbay.py --traffic pems-bay.h5 --adjacency adj_mx_bay.pkl          (run inside the DSTFGCN repository)

Writes data/PEMS-BAY/{train,val,test}.npz and data/graph/adj_PEMS-BAY.pkl.

It follows the authors' pre_process_data.py (generate_seq / generate_from_data): the series is cut chronologically first, 12-step input
windows and the following 12-step targets are cut inside each part, the inputs are standardised with the mean/std of the
training inputs, and the targets stay in the original units. Differences from the authors' script, all for fairness with HGP:
  * the cut points are HGP's: train = first int(0.7 T) rows, validation = next int(0.1 T), test = the rest (the authors use 60/20/20);
  * missing readings (0 in PEMS-BAY) stay 0 in the inputs - the model's loss and metrics treat 0 as missing.
The road graph is DCRNN's adj_mx_bay.pkl, the one HGP uses, stored as [sensor_ids, id_to_index, adj] because train.py reads adj[2].
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from pre_process_data import generate_seq          # the authors' windowing function

ap = argparse.ArgumentParser()
ap.add_argument("--traffic", required=True)
ap.add_argument("--adjacency", required=True)
ap.add_argument("--train_rate", type=float, default=0.7)
ap.add_argument("--val_rate", type=float, default=0.1)
ap.add_argument("--sensors", type=int, default=None, help="keep the first K sensors (smoke test only)")
ap.add_argument("--steps", type=int, default=None, help="keep the first K time steps (smoke test only)")
a = ap.parse_args()

table = pd.read_hdf(a.traffic)
with open(a.adjacency, "rb") as f:
    content = pickle.load(f, encoding="latin1")
ids, id_to_index, adj = content[0], content[1], np.asarray(content[2], dtype=np.float32)
if a.sensors:
    table, adj, ids = table.iloc[:, :a.sensors], adj[:a.sensors, :a.sensors], ids[:a.sensors]
if a.steps:
    table = table.iloc[:a.steps]
data = np.nan_to_num(table.values.astype(np.float32), nan=0.0)[:, :, None]       # [T, N, 1]
T, N, _ = data.shape

train_end = int(T * a.train_rate)
val_end = train_end + int(T * a.val_rate)
parts = {"train": data[:train_end], "val": data[train_end:val_end], "test": data[val_end:]}

out = Path("data/PEMS-BAY")
out.mkdir(parents=True, exist_ok=True)
mean = std = None
for name, part in parts.items():
    x, y = generate_seq(part, 12, 12)                                    # x, y: [windows, 12, N, 1]
    if mean is None:                                                     # as the authors: statistics of the training inputs
        mean, std = x.mean(), x.std()
    np.savez_compressed(out / f"{name}.npz", x=((x - mean) / std).astype(np.float32), y=y.astype(np.float32))
    print(f"{name}: rows {len(part)}, windows {len(x)}, x {x.shape}")

Path("data/graph").mkdir(parents=True, exist_ok=True)
with open("data/graph/adj_PEMS-BAY.pkl", "wb") as f:
    pickle.dump([ids, id_to_index, adj], f)
print(f"{T} steps x {N} sensors, {(data == 0).mean():.4%} missing, train mean {mean:.3f} std {std:.3f}, {(adj != 0).sum()} road links")
