import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgp.config import ModelConfig
from hgp.data import window_index
from hgp.graphs.build import row_normalise, to_sparse
from hgp.metrics import masked_mae
from hgp.model import HICP, count_parameters

H, N, B, T = (3, 6, 12), 7, 2, 12


def graphs(seed):
    rng = np.random.default_rng(seed)
    return {h: to_sparse(row_normalise((rng.random((N, N)) < 0.5) * rng.random((N, N))), "cpu") for h in H}


def model(**kw):
    torch.manual_seed(0)
    return HICP(5, H, ModelConfig(hidden=16, horizon_embed=4, dropout=0.0, **kw))


def test_shapes_parameter_parity_and_gate_readout():
    flow, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    pred, gates = model()(flow, tf, graphs(0))
    assert pred.shape == (B, 3, N) and all(0 <= float(g) <= 1 for g in gates.values())
    assert len({count_parameters(model()), count_parameters(model(use_graph=False)), count_parameters(model(use_gate=False))}) == 1


def test_no_graph_branches_reduce_to_the_per_node_model():
    flow, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    m = model()
    base, _ = m(flow, tf, {h: None for h in H})
    g = graphs(0)
    g[6] = None
    mixed, gates = m(flow, tf, g)
    assert torch.allclose(mixed[:, 1], base[:, 1]) and float(gates[6]) == 0
    assert not torch.allclose(mixed[:, 0], base[:, 0])
    floor = model(use_graph=False)
    a, _ = floor(flow, tf, graphs(0))
    b, _ = floor(flow, tf, graphs(1))
    assert torch.allclose(a, b)


def test_window_index_matches_brute_force():
    rng = np.random.default_rng(0)
    rows, seq_len = 400, 12
    input_ok = rng.random(rows) > 0.05
    target_ok = rng.random(rows) > 0.1
    starts, mask = window_index(input_ok, target_ok, seq_len, H)
    for s, m in zip(starts, mask):
        assert input_ok[s:s + seq_len].all()
        for k, h in enumerate(H):
            row = s + seq_len + h - 1
            assert m[k] == (row < rows and target_ok[row])
    expected = [s for s in range(rows - seq_len) if input_ok[s:s + seq_len].all()
                and any(s + seq_len + h - 1 < rows and target_ok[s + seq_len + h - 1] for h in H)]
    assert starts.tolist() == expected


def test_masked_mae():
    pred, target = torch.zeros(2, 3, 4), torch.ones(2, 3, 4)
    target[:, 2] = float("nan")
    mask = torch.tensor([[True, True, False], [True, False, False]])
    assert torch.isclose(masked_mae(pred, target, mask), torch.tensor(1.0))
