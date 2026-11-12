import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgp.config import GraphConfig
from hgp.graphs import (Assignment, assemble, candidate_pool, fit_var, impulse_responses, knockoff_statistics,
                        knockoff_threshold, row_normalise, save_graph, select_sources, var_top_k)


def ring_road(n):
    road = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        road[i, (i + 1) % n] = road[(i + 1) % n, i] = 1.0
    return road


def test_var_recovers_coefficients_on_simulated_data():
    rng = np.random.default_rng(0)
    n, T = 8, 6000
    road = ring_road(n)
    A1 = np.zeros((n, n))
    for i in range(n):
        A1[i, i], A1[i, (i + 1) % n] = 0.5, 0.3
    x = np.zeros((T, n))
    for t in range(1, T):
        x[t] = A1 @ x[t - 1] + rng.standard_normal(n)
    coef = fit_var(x, road, lag=1, ridge=1e-3)
    assert np.allclose(coef[0], A1, atol=0.05)
    assert np.all(coef[0][(road == 0) & ~np.eye(n, dtype=bool)] == 0)
    psi = impulse_responses(coef, (1, 2, 3))
    assert np.allclose(psi[2], coef[0] @ coef[0]) and np.allclose(psi[3], coef[0] @ coef[0] @ coef[0])


def test_candidate_pool_is_two_hop_neighbourhood():
    pool = candidate_pool(ring_road(10))
    assert pool[0, 1] and pool[0, 2] and not pool[0, 3] and not pool[0, 0]
    assert pool.sum(1).tolist() == [4] * 10


def test_knockoff_threshold_and_selection():
    W = np.array([5.0, 4.0, 3.0, -0.5, 2.0, -0.2, 0.0])
    t = knockoff_threshold(W, q=0.5)                       # t = 0.5: (1 + 1) / 4 <= 0.5
    assert t == 0.5
    assert knockoff_threshold(np.array([-1.0, -2.0, 0.5]), 0.2) == np.inf
    groups = np.array([0, 0, 1, 2, 3, 4, 5])
    W_grp = np.array([5.0, 5.0, 3.0, -0.5, 2.0, -0.2, 0.0])
    W_ind = np.array([1.0, 2.0, 3.0, -0.5, 2.0, -0.2, 0.0])
    keep = select_sources(W_grp, W_ind, groups, q=0.5, max_sources=25, ind_scale=3.0)
    assert sorted(keep.tolist()) == [1, 2, 4]                 # one member per selected group, the one with larger W_ind
    assert len(select_sources(W_grp, W_ind, groups, 0.5, max_sources=2, ind_scale=3.0)) == 2


def test_knockoff_statistics_find_true_sources():
    rng = np.random.default_rng(1)
    T, n, h, true = 20000, 21, 3, [1, 2, 3, 4, 5, 6]
    x = rng.standard_normal((T, n))
    for j in true:
        x[h:, 0] += 0.5 * x[:-h, j]                          # sources 1..6 drive target 0 with lag h
    cfg = GraphConfig(min_rows=500, group_corr=0.5)
    cand = np.arange(1, n)
    W_grp, W_ind, groups, gamma = knockoff_statistics(x, 0, cand, h, T, cfg, rng)
    assert 0 < gamma <= 1
    assert all(W_grp[j - 1] > 0 for j in true)
    assert np.sort(W_grp)[-len(true):].min() > np.sort(W_grp)[:-len(true)].max()
    keep = cand[select_sources(W_grp, W_ind, groups, 0.3, 25, float(np.abs(W_ind).max()))]
    assert set(true) <= set(keep.tolist())
    false = len(set(keep.tolist()) - set(true))
    assert false / len(keep) <= 0.45                                 # within the q = 0.3 guarantee for one draw


def test_assemble_weights_and_assignments(tmp_path):
    n = 6
    rng = np.random.default_rng(0)
    effect = rng.random((n, n)) * (rng.random((n, n)) < 0.5)
    rho = rng.random((n, n)) * 0.1
    selected = rng.random((n, n)) < 0.5
    reference = var_top_k(rng.random((n, n)), 2)
    g, info = assemble(selected, effect, rho, reference)
    assert np.all(np.diag(g) == 0) and np.all(g >= 0)
    assert np.isclose(g[g > 0].mean(), reference[reference > 0].mean(), rtol=1e-5)
    assert info["edges"] == info["var_supported"] + info["added"]
    for h in (3, 6, 12):
        save_graph(tmp_path / f"G_h{h}.npz", g * h, np.arange(n), h, 100)
    fam = Assignment.build("family", (3, 6, 12), tmp_path)
    assert np.allclose(fam.graphs[6], g * 6)
    sh = Assignment.build("shuffled", (3, 6, 12), tmp_path, seed=1)
    assert all(sh.source[h] != f"G_{h}" for h in (3, 6, 12))
    rp = Assignment.build("road_power", (3, 6, 12), tmp_path, ring_road(n))
    assert rp.source == {3: "road^1", 6: "road^2", 12: "road^4"}
    assert Assignment.build("none", (3, 6, 12), tmp_path).graphs[3] is None
    normalised = row_normalise(g)
    assert np.allclose(normalised.sum(1)[(g > 0).any(1)], 1.0)
