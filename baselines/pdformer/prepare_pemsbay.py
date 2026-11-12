"""Convert PEMS-BAY (DCRNN release) into the LibCity atomic files PDFormer reads. Data preparation only; no model code is touched.

    python prepare_pemsbay.py --traffic pems-bay.h5 --adjacency adj_mx_bay.pkl --name PEMSBAY

Writes raw_data/<name>/{<name>.geo, <name>.rel, <name>.dyna, config.json}.
Values are the raw speeds (mph) with missing readings left as 0, which PDFormer's masked loss and metrics treat as missing.
The road links are the non-zero entries of the DCRNN adjacency (the same road graph HGP uses); bidir=true in the model config
symmetrises them.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--traffic", required=True)
ap.add_argument("--adjacency", required=True)
ap.add_argument("--name", default="PEMSBAY")
ap.add_argument("--sensors", type=int, default=None, help="keep the first K sensors (smoke test only)")
ap.add_argument("--steps", type=int, default=None, help="keep the first K time steps (smoke test only)")
a = ap.parse_args()

table = pd.read_hdf(a.traffic)
with open(a.adjacency, "rb") as f:
    content = pickle.load(f, encoding="latin1")
adj = np.asarray(content[2], dtype=np.float32)
if a.sensors:
    table, adj = table.iloc[:, :a.sensors], adj[:a.sensors, :a.sensors]
if a.steps:
    table = table.iloc[:a.steps]
T, N = table.shape
speed = np.nan_to_num(table.values.astype(np.float32), nan=0.0)

out = Path("raw_data") / a.name
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"geo_id": range(N), "type": "Point", "coordinates": "[]"}).to_csv(out / f"{a.name}.geo", index=False)
src, dst = np.nonzero(adj)
pd.DataFrame({"rel_id": range(len(src)), "type": "geo", "origin_id": src, "destination_id": dst,
              "cost": adj[src, dst]}).to_csv(out / f"{a.name}.rel", index=False)
times = np.array([t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in table.index])
with open(out / f"{a.name}.dyna", "w", newline="") as f:
    f.write("dyna_id,type,time,entity_id,traffic_speed\n")
    k = 0
    for j in range(N):
        chunk = pd.DataFrame({"dyna_id": np.arange(k, k + T), "type": "state", "time": times, "entity_id": j,
                              "traffic_speed": speed[:, j]})
        chunk.to_csv(f, header=False, index=False)
        k += T
json.dump({"geo": {"including_types": ["Point"], "Point": {}},
           "rel": {"including_types": ["geo"], "geo": {"cost": "num"}},
           "dyna": {"including_types": ["state"], "state": {"entity_id": "geo_id", "traffic_speed": "num"}},
           "info": {"data_col": ["traffic_speed"], "weight_col": "cost", "data_files": [a.name], "geo_file": a.name,
                    "rel_file": a.name, "output_dim": 1, "time_intervals": 300, "init_weight_inf_or_zero": "zero",
                    "set_weight_link_or_dist": "link", "calculate_weight_adj": False, "weight_adj_epsilon": 0}},
          open(out / "config.json", "w"), indent=2)
print(f"{out}: {T} steps x {N} sensors, {len(src)} road links, {(speed == 0).mean():.4%} missing")
