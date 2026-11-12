"""
Data access for HGP.

The prepared dataset (scripts/prepare_data.py) is one .npz file with
    flow        float32 [T, N]  observed flow, NaN where missing
    inputs      float32 [T, N]  model inputs: flow with short gaps filled, NaN where nothing could be filled
    timestamps  int64   [T]     nanoseconds since the epoch, 5-minute grid
    sensors     int64   [N]     sensor indices in the source dataset
    road        float32 [N, N]  road adjacency (road[i, j] != 0: a road link between i and j)
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig

STEPS_PER_DAY = 288
NS_PER_STEP = 5 * 60 * 10**9


@dataclass
class Split:
    name: str
    inputs: np.ndarray       # [T, N] standardised model inputs
    target_ok: np.ndarray    # [T] every sensor observed at this row
    input_ok: np.ndarray     # [T] every sensor finite in `inputs`
    time_feat: np.ndarray    # [T, 4] sin/cos of time of day and day of week
    offset: int              # row index of this split's first step in the full series
    timestamps: np.ndarray
    observed: np.ndarray     # [T, N] original observation mask


@dataclass
class Data:
    train: Split
    val: Split
    test: Split
    sensors: np.ndarray
    road: np.ndarray
    mean: np.ndarray         # [N] per-sensor standardisation from observed training rows
    std: np.ndarray
    train_end: int
    val_end: int

    @property
    def num_nodes(self) -> int:
        return len(self.sensors)

    def destandardise(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def time_features(timestamps: np.ndarray) -> np.ndarray:
    steps = np.asarray(timestamps, dtype=np.int64) // NS_PER_STEP
    tod = (steps % STEPS_PER_DAY) / STEPS_PER_DAY
    dow = ((steps // STEPS_PER_DAY + 3) % 7) / 7.0          # 1970-01-01 was a Thursday
    angle = 2 * np.pi
    return np.stack([np.sin(angle * tod), np.cos(angle * tod), np.sin(angle * dow), np.cos(angle * dow)], 1).astype(np.float32)


def load(config: DataConfig) -> Data:
    with np.load(config.path) as z:
        flow, inputs, timestamps = z["flow"], z["inputs"], z["timestamps"]
        sensors, road = z["sensors"], z["road"]
    T = len(flow)
    train_end = int(T * config.train_rate)
    val_end = train_end + int(T * config.val_rate)

    mean = np.nanmean(flow[:train_end], axis=0)
    std = np.nanstd(flow[:train_end], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std >= config.min_std), std, 1.0).astype(np.float32)

    def split(name, lo, hi):
        x = ((inputs[lo:hi] - mean) / std).astype(np.float32)
        observed = np.isfinite(flow[lo:hi])
        return Split(name, x, observed.all(1), np.isfinite(x).all(1), time_features(timestamps[lo:hi]), lo,
                     timestamps[lo:hi], observed)

    return Data(split("train", 0, train_end), split("val", train_end, val_end), split("test", val_end, T),
                sensors, road, mean, std, train_end, val_end)


def window_index(input_ok: np.ndarray, target_ok: np.ndarray, seq_len: int, horizons: tuple[int, ...]):
    """
    Window s covers input rows s .. s+seq_len-1 and predicts rows s+seq_len+h-1 for every horizon h.
    A window is kept when its inputs are finite and at least one horizon's target row is fully observed;
    mask[w, k] marks the observed horizons of window w.
    """
    num = len(input_ok) - seq_len - min(horizons) + 1
    bad = np.concatenate([[0], np.cumsum(~input_ok)])
    inputs_bad = (bad[seq_len:seq_len + num] - bad[:num]) > 0
    padded = np.concatenate([target_ok, np.zeros(max(horizons), bool)])
    mask = np.stack([padded[seq_len + h - 1: seq_len + h - 1 + num] for h in horizons], 1)
    starts = np.flatnonzero(~inputs_bad & mask.any(1))
    return starts, mask[starts]


class Windows(Dataset):
    """flow [T, N], time features [T, 4], targets [H, N] (NaN where unobserved), mask [H], window start."""

    def __init__(self, split: Split, seq_len: int, horizons: tuple[int, ...], starts=None):
        self.x, self.tf, self.seq_len, self.horizons = split.inputs, split.time_feat, seq_len, tuple(horizons)
        if starts is None:
            self.starts, self.mask = window_index(split.input_ok, split.target_ok, seq_len, self.horizons)
        else:
            self.starts = np.asarray(starts)
            padded = np.concatenate([split.target_ok, np.zeros(max(horizons), bool)])
            self.mask = np.stack([padded[self.starts + seq_len + h - 1] for h in self.horizons], 1)
        self.offsets = np.array([seq_len + h - 1 for h in self.horizons])

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        rows = np.minimum(s + self.offsets, len(self.x) - 1)
        y = np.where(self.mask[i][:, None], self.x[rows], np.nan).astype(np.float32)
        return (torch.from_numpy(self.x[s:s + self.seq_len].copy()), torch.from_numpy(self.tf[s:s + self.seq_len].copy()),
                torch.from_numpy(y), torch.from_numpy(self.mask[i].copy()), s)

    @staticmethod
    def collate(batch):
        x, tf, y, m, s = zip(*batch)
        return torch.stack(x), torch.stack(tf), torch.stack(y), torch.stack(m), torch.as_tensor(s)
