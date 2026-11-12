"""Test metrics of the trained DSTFGCN in HGP's format (hgp/metrics.py), from the authors' best checkpoint.

    python evaluate_hgp_metrics.py --config_path config/PEMS-BAY_hgp.json --device cuda:0 --out results

The authors' train.py reports MAE / MAPE / RMSE averaged over all 12 steps, with MAPE over every non-zero target. HGP reports steps 3 / 6 / 12
(15 / 30 / 60 min), MAPE over targets >= 1, and averages the three horizons. Here a test window counts at a horizon when every sensor is
observed (> 0) at that step - HGP's rule - and the metrics are computed exactly as in hgp/metrics.py.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import util
from engine import trainer

ap = argparse.ArgumentParser()
ap.add_argument("--config_path", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default="results")
ap.add_argument("--mape_min", type=float, default=1.0)
a = ap.parse_args()

config = json.load(open(a.config_path))
data = util.load_dataset(config["data"], config["N_t"], config["batch_size"], config["batch_size"], config["batch_size"])
adj = pickle.load(open(config["adjacency matrix"], "rb"), encoding="latin1")[2]
device = torch.device(a.device)
model = trainer(config, data["scaler"], adj, a.device).model
ckpt = sorted(glob.glob(config["save"] + f"DSTFGCN_exp{config['expid']}_best_*.pth"), key=os.path.getmtime)[-1]
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
print("checkpoint", ckpt)

realy = torch.Tensor(data["y_test"]).to(device).transpose(1, 3)[:, 0, :, :]          # [W, N, 12]
outputs = []
for x, y, d in data["test_loader"].get_iterator():
    with torch.no_grad():
        outputs.append(model(torch.Tensor(x).to(device).transpose(1, 3), d).transpose(1, 3).squeeze(1))
yhat = torch.cat(outputs, 0)[: realy.size(0)]
pred = data["scaler"].inverse_transform(yhat).cpu().numpy().astype(np.float64)        # [W, N, 12]
real = realy.cpu().numpy().astype(np.float64)


def evaluate(t, p):
    err = p - t
    keep = np.abs(t) >= a.mape_min
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape": float(100 * np.mean(np.abs(err[keep]) / np.abs(t[keep]))) if keep.any() else float("nan"), "n": int(len(t))}


rows, saved = {}, {}
for h in (3, 6, 12):
    t, p = real[:, :, h - 1], pred[:, :, h - 1]
    ok = (t > 0).all(axis=1)
    rows[h] = evaluate(t[ok], p[ok])
    saved[f"targets_h{h}"], saved[f"predictions_h{h}"] = t[ok].astype(np.float32), p[ok].astype(np.float32)
    print(f"h={h:2d} ({5 * h:2d} min)  MAE {rows[h]['mae']:.3f}  RMSE {rows[h]['rmse']:.3f}  MAPE {rows[h]['mape']:.2f}%  windows {rows[h]['n']}")
rows["mean"] = {k: float(np.mean([rows[h][k] for h in (3, 6, 12)])) for k in ("mae", "rmse", "mape")}
rows["error_growth_rmse"] = rows[12]["rmse"] / rows[3]["rmse"]
print("mean of h3/h6/h12:", rows["mean"], "| RMSE growth h12/h3:", round(rows["error_growth_rmse"], 3))
valid = real > 0
e = (pred - real)[valid]
rows["all_12_steps"] = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean()))}
print("pooled over all 12 steps (entries > 0):", rows["all_12_steps"])
os.makedirs(a.out, exist_ok=True)
json.dump(rows, open(os.path.join(a.out, "test_metrics.json"), "w"), indent=2)
np.savez_compressed(os.path.join(a.out, "test_predictions.npz"), **saved)
