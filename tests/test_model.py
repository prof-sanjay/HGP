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
    return HICP(5, H, ModelConfig(hidden=16, horizon_embed=4, dropout=0.0, **kw), num_nodes=N)


def road():
    r = np.zeros((N, N), dtype=np.float32)
    for i in range(N - 1):
        r[i, i + 1] = r[i + 1, i] = 1.0
    return to_sparse(row_normalise(r), "cpu")


def test_shapes_parameter_parity_and_gate_readout():
    flow, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    pred, gates = model()(flow, tf, graphs(0))
    assert pred.shape == (B, 3, N) and all(0 <= float(g["causal"]) <= 1 and set(g) == {"causal"} for g in gates.values())
    assert len({count_parameters(model()), count_parameters(model(use_graph=False)), count_parameters(model(use_gate=False))}) == 1


def test_single_branch_gate_is_the_sigmoid_gate():
    """With the causal branch alone, softmax([0, l]) = sigmoid(l): H = a S + (1 - a) Z exactly as in the original HICP."""
    flow, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    m = model()
    g = graphs(0)
    with torch.no_grad():
        Z = m.encoder(flow, tf)
        S = m.propagation[0](Z, g[3])
        e = m.gate.embed.weight[0].expand(B, N, -1)
        a = torch.sigmoid(m.gate.mlp(torch.cat([Z, S, e], -1)))
        expected = m.heads[0](a * S + (1 - a) * Z).squeeze(-1)
        pred, _ = m(flow, tf, g)
    assert torch.allclose(pred[:, 0], expected, atol=1e-6)


def test_no_graph_branches_reduce_to_the_per_node_model():
    flow, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    m = model()
    base, _ = m(flow, tf, {h: None for h in H})
    g = graphs(0)
    g[6] = None
    mixed, gates = m(flow, tf, g)
    assert torch.allclose(mixed[:, 1], base[:, 1]) and float(gates[6]["causal"]) == 0
    assert not torch.allclose(mixed[:, 0], base[:, 0])
    floor = model(use_graph=False)
    a, _ = floor(flow, tf, graphs(0))
    b, _ = floor(flow, tf, graphs(1))
    assert torch.allclose(a, b)


def test_semantic_graph_is_row_stochastic_top_k_without_self_loops():
    torch.manual_seed(0)
    from hgp.model import SemanticGraph
    A = SemanticGraph(N, 4, top_k=3)()
    assert A.shape == (N, N) and torch.allclose(A.sum(1), torch.ones(N), atol=1e-6)
    assert ((A > 0).sum(1) == 3).all() and float(A.diagonal().abs().max()) == 0
    A.sum().backward()      # the embeddings receive gradient through the kept entries


def test_multi_graph_branches_mix_convexly_and_keep_parity():
    flow, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    kw = dict(physical=True, semantic=True, semantic_dim=4, semantic_top_k=3)
    m = model(**kw)
    assert m.branches == ["causal", "physical", "semantic"]
    pred, gates = m(flow, tf, graphs(0), physical=road())
    assert pred.shape == (B, 3, N)
    for g in gates.values():
        assert set(g) == {"causal", "physical", "semantic"} and 0 < sum(float(v) for v in g.values()) <= 1 + 1e-6
    # the same parameter count for none / family / g12_all when the extra branches are on
    assert count_parameters(m) == count_parameters(model(use_graph=False, **kw)) > count_parameters(model())
    # without a causal graph the causal branch gets no weight, the others still work
    none_pred, none_gates = model(use_graph=False, **kw)(flow, tf, graphs(0), physical=road())
    assert all(float(g["causal"]) == 0 and float(g["physical"]) > 0 and float(g["semantic"]) > 0 for g in none_gates.values())
    # the convex weights sum to one over [self, causal, physical, semantic]
    with torch.no_grad():
        Z = m.encoder(flow, tf)
        A = m.semantic_graph()
        S = [m.propagation[0](Z, graphs(0)[3]), m.physical_propagation[0](Z, road()), m.semantic_propagation[0](Z, A)]
        alpha = m.gate(Z, S, 0, [True, True, True])
    assert alpha.shape == (B, N, 4, 16) and torch.allclose(alpha.sum(2), torch.ones(B, N, 16), atol=1e-6)


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
