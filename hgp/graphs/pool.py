"""
Candidate pool and the partial-correlation score used to weight edges outside the VAR support.

Two sensors are candidates for each other when they share a local unit U_k = {k} + road neighbours of k
(both directions), i.e. when they are within two undirected road hops.
"""

import numpy as np


def candidate_pool(road: np.ndarray) -> np.ndarray:
    """Boolean [N, N]: pool[i, j] is True when i and j share a local road unit; diagonal False."""
    N = road.shape[0]
    unit = ((road != 0) | (road != 0).T | np.eye(N, dtype=bool)).astype(np.int32)
    pool = (unit @ unit) > 0
    np.fill_diagonal(pool, False)
    return pool


def partial_correlation(flow: np.ndarray, pool: np.ndarray, horizon: int, shrink: float = 0.02, min_rows: int = 1000) -> np.ndarray:
    """
    score[i, j] = partial correlation of x_i(t+h) with x_j(t), given x_i(t) and the other pool sources of i,
    from the shrunk inverse correlation matrix of [x_i(t+h), x_i(t), x_pool(t)] on complete training rows.
    """
    T, N = flow.shape
    score = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        cols = np.flatnonzero(pool[i])
        if len(cols) == 0:
            continue
        v = np.column_stack([flow[horizon:, i], flow[:T - horizon, i], flow[:T - horizon][:, cols]])
        v = v[np.isfinite(v).all(axis=1)]
        if len(v) < min_rows:
            continue
        corr = np.nan_to_num(np.corrcoef(v, rowvar=False))
        omega = np.linalg.inv((1 - shrink) * corr + shrink * np.eye(len(corr)))
        d = np.sqrt(np.abs(np.diag(omega)))
        score[i, cols] = np.abs(omega[0, 2:]) / (d[0] * d[2:])
    return score
