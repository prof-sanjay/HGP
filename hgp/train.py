import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, TrainConfig
from .data import Data, Windows
from .graphs.build import Assignment
from .metrics import horizon_metrics, masked_mae
from .model import HICP, count_parameters


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, data: Data, data_config: DataConfig, config: TrainConfig, assignment: Assignment,
                 device: torch.device | None = None, max_windows: int | None = None):
        self.data, self.dc, self.config = data, data_config, config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(config.seed)
        self.graphs = assignment.to_sparse(self.device)
        self.assignment = assignment
        self.model = HICP(5, data_config.horizons, config.model).to(self.device)
        self.loaders = {}
        for split in (data.train, data.val, data.test):
            ds = Windows(split, data_config.seq_len, data_config.horizons)
            if max_windows:
                ds = Windows(split, data_config.seq_len, data_config.horizons, ds.starts[:max_windows])
            self.loaders[split.name] = DataLoader(ds, config.batch_size, shuffle=split.name == "train",
                                                  collate_fn=Windows.collate, generator=torch.Generator().manual_seed(config.seed))
        self.run_dir = config.run_dir()
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _schedule(self, optimizer):
        steps = len(self.loaders["train"])
        total, warm = self.config.epochs * steps, max(1, self.config.warmup_epochs * steps)

        def factor(step):
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def train_epoch(self, optimizer, scheduler) -> float:
        self.model.train()
        total = count = 0
        for flow, tf, y, m, _ in self.loaders["train"]:
            flow, tf, y, m = (t.to(self.device) for t in (flow, tf, y, m))
            pred, _ = self.model(flow, tf, self.graphs)
            loss = masked_mae(pred, y, m)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            optimizer.step()
            scheduler.step()
            total += float(loss) * len(flow)
            count += len(flow)
        return total / count

    @torch.no_grad()
    def predict(self, name: str):
        """Per horizon: targets and predictions in vehicles (observed windows only), window starts, mean gate."""
        self.model.eval()
        H = self.dc.horizons
        targets, preds, starts = ({h: [] for h in H} for _ in range(3))
        gate = {h: 0.0 for h in H}
        mean, std = (torch.as_tensor(v, device=self.device) for v in (self.data.mean, self.data.std))
        n = 0
        for flow, tf, y, m, s in self.loaders[name]:
            flow, tf, y = (t.to(self.device) for t in (flow, tf, y))
            pred, g = self.model(flow, tf, self.graphs)
            pred, y = pred * std + mean, y * std + mean
            for k, h in enumerate(H):
                keep = m[:, k].numpy()
                if keep.any():
                    preds[h].append(pred[keep, k].cpu().numpy())
                    targets[h].append(y[keep, k].cpu().numpy())
                    starts[h].append(s.numpy()[keep])
                gate[h] += float(g[h]) * len(flow)
            n += len(flow)
        cat = lambda d: {h: np.concatenate(v) if v else np.zeros((0, self.data.num_nodes)) for h, v in d.items()}
        return cat(targets), cat(preds), cat(starts), {h: v / max(n, 1) for h, v in gate.items()}

    def fit(self) -> dict:
        cfg = self.config
        optimizer = torch.optim.AdamW(self.model.parameters(), cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = self._schedule(optimizer)
        best, bad, t0 = {"epoch": 0, "mae": float("inf")}, 0, time.time()
        self._json("config.json", {"train": asdict(cfg), "data": asdict(self.dc), "graphs": self.assignment.summary(),
                                   "parameters": count_parameters(self.model), "device": str(self.device)})
        with open(self.run_dir / "history.csv", "w", newline="") as f:
            writer = None
            for epoch in range(1, cfg.epochs + 1):
                t1 = time.time()
                train_loss = self.train_epoch(optimizer, scheduler)
                targets, preds, _, gate = self.predict("val")
                val = horizon_metrics(targets, preds, self.dc.mape_min_flow)
                row = {"epoch": epoch, "train_loss": train_loss, "val_mae": val["mean"]["mae"], "val_rmse": val["mean"]["rmse"],
                       **{f"val_mae_h{h}": val[h]["mae"] for h in self.dc.horizons},
                       **{f"gate_h{h}": gate[h] for h in self.dc.horizons},
                       "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t1}
                writer = writer or csv.DictWriter(f, fieldnames=list(row))
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                improved = val["mean"]["mae"] < best["mae"]
                if improved:
                    best, bad = {"epoch": epoch, "mae": val["mean"]["mae"], "val": val, "gate": gate}, 0
                    torch.save(self.model.state_dict(), self.run_dir / "checkpoints" / "best.pt")
                else:
                    bad += 1
                print(f"epoch {epoch:3d}  loss {train_loss:.4f}  val MAE {val['mean']['mae']:.3f}  "
                      f"gate {' '.join(f'{gate[h]:.2f}' for h in self.dc.horizons)}  {row['seconds']:.0f}s{'  *' if improved else ''}", flush=True)
                if bad >= cfg.patience:
                    break

        self.model.load_state_dict(torch.load(self.run_dir / "checkpoints" / "best.pt", map_location=self.device))
        targets, preds, starts, gate = self.predict("test")
        test = horizon_metrics(targets, preds, self.dc.mape_min_flow)
        self._save_predictions(targets, preds, starts)
        results = {"run": self.run_dir.name, "parameters": count_parameters(self.model), "best_epoch": best["epoch"],
                   "epochs": epoch, "validation": best["val"], "test": test, "test_gate": gate,
                   "training_seconds": time.time() - t0, "device": str(self.device)}
        self._json("results.json", results)
        print("test  " + "  ".join(f"h{h}: RMSE {test[h]['rmse']:.3f} MAE {test[h]['mae']:.3f}" for h in self.dc.horizons))
        return results

    def _save_predictions(self, targets, preds, starts) -> None:
        split, T = self.data.test, self.dc.seq_len
        for h in self.dc.horizons:
            rows = starts[h] + T + h - 1
            np.savez_compressed(self.run_dir / f"test_predictions_h{h}.npz", targets=targets[h].astype(np.float32),
                                predictions=preds[h].astype(np.float32), rows=(split.offset + rows).astype(np.int64),
                                timestamps=split.timestamps[rows], sensors=self.data.sensors,
                                input_fully_observed=np.array([split.observed[s:s + T].all() for s in starts[h]]))

    def _json(self, name: str, obj) -> None:
        (self.run_dir / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
