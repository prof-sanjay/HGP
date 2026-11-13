import numpy as np
import torch


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred, target [B, H, N]; mask [B, H]. Horizons are averaged with equal weight."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    err = ((pred - torch.nan_to_num(target)).abs() * m).sum((0, 2))
    count = (m.sum((0, 2)) * pred.shape[-1]).clamp_min(1.0)
    present = (mask.sum(0) > 0).to(pred.dtype)
    return ((err / count) * present).sum() / present.sum().clamp_min(1.0)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, mape_min_flow: float = 10.0) -> dict:
    y_true, y_pred = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    err = y_pred - y_true
    keep = np.abs(y_true) >= mape_min_flow
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(100 * np.mean(np.abs(err[keep]) / np.abs(y_true[keep]))) if keep.any() else float("nan"),
        "n": int(y_true.shape[0]),
    }


def horizon_metrics(targets: dict, predictions: dict, mape_min_flow: float = 10.0) -> dict:
    out = {int(h): evaluate(targets[h], predictions[h], mape_min_flow) for h in sorted(targets)}
    hs = sorted(out)
    main = [h for h in hs if h in (3, 6, 12)] or hs      # "mean" stays the mean over 15 / 30 / 60 min, also when h=1 is added
    out["mean"] = {k: float(np.mean([out[h][k] for h in main])) for k in ("mae", "rmse", "mape")}
    if main != hs:
        out["mean_all"] = {k: float(np.mean([out[h][k] for h in hs])) for k in ("mae", "rmse", "mape")}
    first = 3 if 3 in hs else hs[0]       # h12 / h3 whenever h=3 is present, so adding h=1 keeps the definition
    out["error_growth_rmse"] = out[hs[-1]]["rmse"] / out[first]["rmse"]
    return out
