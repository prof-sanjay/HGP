"""Runs the authors' train.py unchanged, with HGP's optimiser and learning-rate schedule.

engine.py hard-codes Adam and a MultiStepLR that train.py steps once per epoch. HGP trains with AdamW, one warm-up epoch and cosine
decay to zero, stepped once per batch. This wrapper swaps exactly those two objects before train.py runs; the model, the loss
(masked Huber), gradient clipping at 5, the data loader and the model selection (best validation loss, early stopping) are the authors' own.

    python run_hgp_protocol.py --config_path config/PEMS-BAY_hgp.json --device cuda:0
"""
import math
import runpy
import sys

import torch

sys.path.insert(0, ".")
import util
import engine

steps = {}
_load = util.load_dataset


def load_dataset(*args, **kwargs):
    data = _load(*args, **kwargs)
    steps["per_epoch"] = data["train_loader"].num_batch
    return data


util.load_dataset = load_dataset


class PerBatchSchedule:
    """HGP's schedule (hgp/train.py): linear warm-up for `warmup_epochs`, then cosine decay to 0."""

    def __init__(self, optimizer, max_epoch, warmup_epochs=1):
        total = max_epoch * steps["per_epoch"]
        warm = max(1, warmup_epochs * steps["per_epoch"])

        def factor(step):
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

        self.inner = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def step(self):          # train.py calls this once per epoch: ignored, the schedule advances per batch
        pass


_init, _train = engine.trainer.__init__, engine.trainer.train


def init(self, config, scaler, adjacency_matrix, device):
    torch.optim.Adam = lambda params, lr, weight_decay: torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    torch.optim.lr_scheduler.MultiStepLR = lambda optimizer, **kw: PerBatchSchedule(optimizer, config["max_epoch"])
    _init(self, config, scaler, adjacency_matrix, device)


def train(self, input, real_val, time_i):
    result = _train(self, input, real_val, time_i)
    self.scheduler.inner.step()
    return result


engine.trainer.__init__, engine.trainer.train = init, train
runpy.run_path("train.py", run_name="__main__")
