"""Build the one-off dataset caches (DTW matrix, windowed arrays, KShape pattern keys) through PDFormer's own data pipeline,
without training. Needs no GPU, so it can run in a CPU-only session; run_model.py then finds the caches and starts training at once.

    python precompute_keys.py --dataset PEMSBAY --config_file PEMSBAY_hgp
"""
import argparse

from libcity.config import ConfigParser
import libcity.data.dataset as datasets
from libcity.data import get_dataset

from pdformer_lowmem import PDFormerLowMemDataset

datasets.PDFormerDataset = PDFormerLowMemDataset

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--config_file", required=True)
a = ap.parse_args()
config = ConfigParser("traffic_state_pred", "PDFormer", a.dataset, a.config_file, True, True, {"gpu": False})
dataset = get_dataset(config)
dataset.get_data()
print("caches written to libcity/cache/dataset_cache/")
