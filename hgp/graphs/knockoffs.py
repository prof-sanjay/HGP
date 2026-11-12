"""
Group-knockoff selection of the sources of one target at horizon h.

For target i, response r = x_i(t) with its own history x_i(t-h), ..., x_i(t-h-L+1) regressed out; candidates
x_j(t-h) for every pool source j, residualised the same way. Second-order Gaussian group knockoffs are drawn
(Dai & Barber), a lasso path is run on [candidates, knockoffs], and the statistic compares the penalty at which a
group enters against the penalty at which its knockoff copy enters. The knockoff+ threshold at level q bounds the
expected false discovery rate of the selected groups by q.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import eigh
from scipy.spatial.distance import squareform
from sklearn.linear_model import lars_path

from ..config import GraphConfig


def detrend(flow: np.ndarray, train_end: int, steps_per_day: int = 288) -> np.ndarray:
    """Remove each sensor's time-of-day mean, estimated on training rows only."""
    out = np.array(flow, dtype=np.float64)
    slot = np.arange(len(flow)) % steps_per_day
    for s in range(steps_per_day):
        rows = np.flatnonzero(slot == s)
        out[rows] -= np.nanmean(out[rows[rows < train_end]], axis=0)
    return out


def make_groups(sigma: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage clusters of the candidates at |correlation| >= threshold."""
    m = len(sigma)
    if m < 2:
        return np.zeros(m, dtype=int)
    dist = 1.0 - np.abs(sigma)
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(np.clip((dist + dist.T) / 2, 0, None), checks=False), method="average")
    return fcluster(link, t=1.0 - threshold, criterion="distance") - 1


def entry_penalty(alphas: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Largest lasso penalty at which each column enters the path (0 if it never enters)."""
    nonzero = coefs != 0
    first = np.argmax(nonzero, axis=1)
    return np.where(nonzero.any(axis=1), alphas[np.minimum(first, len(alphas) - 1)], 0.0)


def knockoff_statistics(flow: np.ndarray, target: int, candidates: np.ndarray, horizon: int, train_end: int,
                        config: GraphConfig, rng: np.random.Generator):
    """
    Returns (W_group, W_individual, groups, gamma) for one target, or None when fewer than config.min_rows clean rows
    exist. W_group is copied to every member of its group; W_individual breaks ties inside a group.
    """
    L = config.self_lags
    rows = np.arange(horizon + L - 1, train_end, config.stride)
    y = flow[rows, target]
    own = np.stack([flow[rows - horizon - l, target] for l in range(L)], axis=1)
    X = flow[rows - horizon][:, candidates]
    ok = np.isfinite(y) & np.isfinite(own).all(axis=1) & np.isfinite(X).all(axis=1)
    if ok.sum() < config.min_rows:
        return None
    y, own, X = y[ok], own[ok], X[ok]
    n, m = X.shape

    design = np.column_stack([np.ones(n), own])
    beta = np.linalg.lstsq(design, np.column_stack([y, X]), rcond=None)[0]
    resid = np.column_stack([y, X]) - design @ beta
    y, X = resid[:, 0], resid[:, 1:]
    X = X / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))

    sigma = (1 - config.shrink) * (X.T @ X) / n + config.shrink * np.eye(m)
    d = np.sqrt(np.diag(sigma))
    sigma, X = sigma / np.outer(d, d), X / d

    groups = make_groups(sigma, config.group_corr)
    block = np.zeros_like(sigma)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        block[np.ix_(idx, idx)] = sigma[np.ix_(idx, idx)]
    gamma = min(1.0, 2.0 * float(eigh(sigma, block, eigvals_only=True)[0]))
    S = gamma * block
    w, U = np.linalg.eigh(sigma)
    inv = (U / w) @ U.T
    V = 2 * S - S @ inv @ S
    v, Vu = np.linalg.eigh((V + V.T) / 2)
    knockoffs = X - X @ inv @ S + rng.standard_normal((n, m)) @ (Vu * np.sqrt(np.clip(v, 1e-12, None))).T

    alphas, _, coefs = lars_path(np.concatenate([X, knockoffs], axis=1), y, method="lasso", max_iter=4 * m)
    Z = entry_penalty(alphas, coefs)
    z, zk = Z[:m], Z[m:]
    W_ind = np.where(z == zk, 0.0, np.maximum(z, zk) * np.sign(z - zk))
    W_grp = np.zeros(m)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        zg, zkg = z[idx].max(), zk[idx].max()
        W_grp[idx] = 0.0 if zg == zkg else max(zg, zkg) * np.sign(zg - zkg)
    return W_grp, W_ind, groups, gamma


def knockoff_threshold(W: np.ndarray, q: float) -> float:
    """knockoff+: the smallest t with (1 + #{W <= -t}) / #{W >= t} <= q (inf when no t qualifies)."""
    for t in np.sort(np.unique(np.abs(W[W != 0]))):
        pos = (W >= t).sum()
        if pos > 0 and (1 + (W <= -t).sum()) / pos <= q:
            return float(t)
    return np.inf


def select_sources(W_grp: np.ndarray, W_ind: np.ndarray, groups: np.ndarray, q: float, max_sources: int,
                   ind_scale: float) -> np.ndarray:
    """Indices of the selected candidates: one representative per group above the threshold, at most max_sources."""
    _, first = np.unique(groups, return_index=True)
    t = knockoff_threshold(W_grp[first], q)
    keep = []
    for g in np.unique(groups[W_grp >= t]):
        members = np.flatnonzero(groups == g)
        keep.append(members[np.argmax(W_ind[members])])
    keep = np.asarray(keep, dtype=int)
    if len(keep) > max_sources:
        order = np.argsort(-(W_grp[keep] + 1e-4 * W_ind[keep] / (ind_scale + 1e-12)))
        keep = keep[order[:max_sources]]
    return keep
