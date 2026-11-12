"""Compare the memory-light loader with PDFormer's original on a small slice: windows, split, padding, scaler must be identical.
    python tests/check_lowmem.py
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import libcity.data.dataset as datasets
from libcity.config import ConfigParser
from pdformer_lowmem import PDFormerLowMemDataset
from libcity.data.dataset.pdformer_dataset import PDFormerDataset

NAME = "PEMSBAYchk"
D = "../../datasets/PeMS-BAY-2022"
subprocess.run([sys.executable, "prepare_pemsbay.py", "--traffic", f"{D}/pems-bay.h5", "--adjacency", f"{D}/adj_mx_bay.pkl",
                "--name", NAME, "--sensors", "30", "--steps", "1500"], check=True)
subprocess.run([sys.executable, "precompute_dtw.py", "--name", NAME, "--workers", "4"], check=True, stdout=subprocess.DEVNULL)
cfg = json.load(open("PEMSBAY_hgp.json")); cfg["max_epoch"] = 1
json.dump(cfg, open(f"{NAME}_cfg.json", "w"))
config = ConfigParser("traffic_state_pred", "PDFormer", NAME, f"{NAME}_cfg", True, True, {"gpu": False})

orig = PDFormerDataset(config)
x_tr, y_tr, x_va, y_va, x_te, y_te = orig._generate_train_val_test()
d = orig.output_dim
scaler = orig._get_scalar("standard", x_tr[..., :d], y_tr[..., :d])
for a, b in ((x_tr, y_tr), (x_va, y_va), (x_te, y_te)):
    a[..., :d] = scaler.transform(a[..., :d]); b[..., :d] = scaler.transform(b[..., :d])

lazy = PDFormerLowMemDataset(config)
os.makedirs("libcity/cache/dataset_cache", exist_ok=True)
np.save(f"libcity/cache/dataset_cache/pattern_keys_kshape_{NAME}_14_3_16_5.npy", np.zeros((16, 3, 1)))   # skip clustering
tr, va, te = lazy.get_data()
print("scaler mean/std  original", scaler.mean, scaler.std, " lazy", lazy.scaler.mean, lazy.scaler.std)
assert abs(scaler.mean - lazy.scaler.mean) < 1e-9 and abs(scaler.std - lazy.scaler.std) < 1e-9
for name, loader, (x, y) in (("train", tr, (x_tr, y_tr)), ("val", va, (x_va, y_va)), ("test", te, (x_te, y_te))):
    n = len(x); bs = config["batch_size"]; padded = n + (bs - n % bs) % bs
    assert len(loader.dataset) == padded, (name, len(loader.dataset), padded)
    worst = 0.0
    for i in range(padded):
        xi, yi = loader.dataset[i]
        j = min(i, n - 1)
        worst = max(worst, np.abs(xi - x[j]).max(), np.abs(yi - y[j]).max())
    print(f"{name}: {n} windows (+{padded - n} padding), max abs difference {worst:.3e}")
    assert worst < 1e-12
print("identical")
