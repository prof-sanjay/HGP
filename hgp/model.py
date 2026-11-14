"""
HICP: horizon-indexed causal propagation, optionally with physical and semantic graph branches.

    Z       = Encoder(X)                              per-node GRU, shared weights, no cross-sensor mixing
    S_h^c   = GELU(G_h Z W_h^c + b_h^c)               one directed hop over the fixed horizon-h causal graph
    S_h^p   = GELU(A_road Z W_h^p + b_h^p)            [--physical]  one hop over the road adjacency (shared graph, per-horizon weights)
    S_h^s   = GELU(A_sem Z W_h^s + b_h^s)             [--semantic]  one hop over a learned similarity graph (shared graph, per-horizon weights)
    A_sem   = softmax_j(E_t E_s^T / sqrt(d_s))        top-k sources per target, learned node embeddings E_t, E_s, no self-loops
    alpha_h = softmax_m([0, MLP([Z, S_h^c, S_h^p, S_h^s, e_h])])   one weight per branch, per node and feature;
                                                       the self branch has a fixed zero logit, absent branches get -inf
    H_h     = alpha_h^0 Z + sum_m alpha_h^m S_h^m       convex mixture of the branches
    y_h     = H_h w_h + b_h                             one head per horizon, all horizons in one forward pass

With only the causal branch, softmax over [0, l] equals sigmoid(l): the gate, the parameter count and the
checkpoints are those of the original HICP (H_h = a_h S_h + (1 - a_h) Z).
"""

import math

import torch
from torch import nn

from .config import ModelConfig

BRANCHES = ("causal", "physical", "semantic")


def propagate(graph: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """graph: [N, N] (sparse or dense) with graph[i, j] the weight of source j for target i; Z: [B, N, d]."""
    B, N, d = Z.shape
    flat = Z.permute(1, 0, 2).reshape(N, B * d)
    out = torch.sparse.mm(graph, flat) if graph.is_sparse else graph @ flat
    return out.reshape(N, B, d).permute(1, 0, 2)


class Encoder(nn.Module):
    def __init__(self, num_features: int, hidden: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(num_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, flow: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        B, T, N = flow.shape
        x = torch.cat([flow.unsqueeze(-1), time_feat.unsqueeze(2).expand(B, T, N, -1)], -1)
        _, h = self.gru(x.permute(0, 2, 1, 3).reshape(B * N, T, -1))
        return self.dropout(h[-1]).reshape(B, N, -1)


class Propagation(nn.Module):
    def __init__(self, hidden: int, two_hop: bool, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2 if two_hop else 1)])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z: torch.Tensor, graph: torch.Tensor | None) -> torch.Tensor:
        if graph is None:
            return torch.zeros_like(Z)
        S = Z
        for layer in self.layers:
            S = self.act(layer(propagate(graph, S)))
        return self.dropout(S)


class SemanticGraph(nn.Module):
    """Learned static similarity graph: A[i, j] = softmax_j(E_t[i] . E_s[j] / sqrt(d)) over the top_k most similar sources j != i."""

    def __init__(self, num_nodes: int, dim: int, top_k: int):
        super().__init__()
        self.target = nn.Parameter(torch.randn(num_nodes, dim) / math.sqrt(dim))
        self.source = nn.Parameter(torch.randn(num_nodes, dim) / math.sqrt(dim))
        self.top_k = min(top_k, num_nodes - 1)

    def forward(self) -> torch.Tensor:
        scores = (self.target @ self.source.T) / math.sqrt(self.target.shape[1])
        scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool, device=scores.device), float("-inf"))
        if self.top_k < len(scores) - 1:
            threshold = scores.topk(self.top_k, dim=1).values[:, -1:]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
        return torch.softmax(scores, dim=1)


class Gate(nn.Module):
    """Softmax over [self, branch_1, ..., branch_M] per node and feature; the self logit is fixed at zero."""

    def __init__(self, hidden: int, num_horizons: int, embed: int, branches: int = 1):
        super().__init__()
        self.branches = branches
        self.embed = nn.Embedding(num_horizons, embed)
        self.mlp = nn.Sequential(nn.Linear((1 + branches) * hidden + embed, hidden), nn.GELU(), nn.Linear(hidden, branches * hidden))

    def forward(self, Z: torch.Tensor, S: list[torch.Tensor], k: int, available: list[bool]) -> torch.Tensor:
        """Returns alpha [B, N, 1 + branches, d]; alpha[..., 0, :] weights Z, alpha[..., m, :] weights S[m - 1]."""
        B, N, d = Z.shape
        e = self.embed.weight[k].expand(B, N, -1)
        logits = self.mlp(torch.cat([Z, *S, e], -1)).view(B, N, self.branches, d)
        logits = torch.cat([logits.new_zeros(B, N, 1, d), logits], 2)
        mask = torch.tensor([False] + [not a for a in available], device=Z.device).view(1, 1, -1, 1)
        return torch.softmax(logits.masked_fill(mask, float("-inf")), dim=2)


class HICP(nn.Module):
    def __init__(self, num_features: int, horizons: tuple[int, ...], config: ModelConfig, num_nodes: int | None = None):
        super().__init__()
        self.horizons = tuple(horizons)
        self.config = config
        self.branches = ["causal"] + [b for b in ("physical", "semantic") if getattr(config, b)]
        self.encoder = Encoder(num_features, config.hidden, config.dropout)
        self.propagation = nn.ModuleList([Propagation(config.hidden, config.two_hop, config.dropout) for _ in horizons])
        if config.physical:
            self.physical_propagation = nn.ModuleList([Propagation(config.hidden, config.two_hop, config.dropout) for _ in horizons])
        if config.semantic:
            if num_nodes is None:
                raise ValueError("the semantic graph needs num_nodes")
            self.semantic_graph = SemanticGraph(num_nodes, config.semantic_dim, config.semantic_top_k)
            self.semantic_propagation = nn.ModuleList([Propagation(config.hidden, config.two_hop, config.dropout) for _ in horizons])
        self.gate = Gate(config.hidden, len(horizons), config.horizon_embed, len(self.branches))
        self.heads = nn.ModuleList([nn.Linear(config.hidden, 1) for _ in horizons])

    def branch_graphs(self, h: int, graphs: dict[int, torch.Tensor | None], physical: torch.Tensor | None,
                      semantic: torch.Tensor | None) -> list[torch.Tensor | None]:
        out = [graphs.get(h) if self.config.use_graph else None]
        if self.config.physical:
            out.append(physical)
        if self.config.semantic:
            out.append(semantic)
        return out

    def forward(self, flow: torch.Tensor, time_feat: torch.Tensor, graphs: dict[int, torch.Tensor | None],
                physical: torch.Tensor | None = None):
        """
        graphs: causal graph per horizon (None = no causal graph); physical: road adjacency (used when config.physical).
        Returns predictions [B, H, N] (standardised) and, per horizon, the mean gate weight of every branch.
        """
        Z = self.encoder(flow, time_feat)
        semantic = self.semantic_graph() if self.config.semantic else None
        preds, gates = [], {}
        for k, h in enumerate(self.horizons):
            branch_graphs = self.branch_graphs(h, graphs, physical, semantic)
            modules = [self.propagation[k]] + [getattr(self, f"{b}_propagation")[k] for b in self.branches[1:]]
            available = [g is not None for g in branch_graphs]
            S = [m(Z, g) for m, g in zip(modules, branch_graphs)]
            if not any(available):
                H, weights = Z, [Z.new_zeros(()) for _ in self.branches]
            elif self.config.use_gate:
                alpha = self.gate(Z, S, k, available)
                H = alpha[:, :, 0] * Z + sum(alpha[:, :, m + 1] * S[m] for m in range(len(S)))
                weights = [alpha[:, :, m + 1].detach().mean() for m in range(len(S))]
            else:
                H = Z + sum(S)
                weights = [Z.new_ones(()) if a else Z.new_zeros(()) for a in available]
            preds.append(self.heads[k](H).squeeze(-1))
            gates[h] = dict(zip(self.branches, weights))
        return torch.stack(preds, 1), gates


class SensorLSTM(nn.Module):
    """Baseline: one LSTM shared by all sensors, each sensor sees only its own 12-step history (flow + time features).

    No road graph, no causal graph, no gate. Same inputs, targets, horizons and loss as HICP; one linear head per horizon
    on the last hidden state of the top layer, so all horizons come out of one forward pass.
    """

    def __init__(self, num_features: int, horizons: tuple[int, ...], config: ModelConfig):
        super().__init__()
        self.horizons = tuple(horizons)
        self.branches = ["causal"]
        self.lstm = nn.LSTM(num_features, config.hidden, config.layers, batch_first=True,
                            dropout=config.dropout if config.layers > 1 else 0.0)
        self.dropout = nn.Dropout(config.dropout)
        self.heads = nn.ModuleList([nn.Linear(config.hidden, 1) for _ in horizons])

    def forward(self, flow: torch.Tensor, time_feat: torch.Tensor, graphs=None, physical=None):
        B, T, N = flow.shape
        x = torch.cat([flow.unsqueeze(-1), time_feat.unsqueeze(2).expand(B, T, N, -1)], -1)
        out, _ = self.lstm(x.permute(0, 2, 1, 3).reshape(B * N, T, -1))
        Z = self.dropout(out[:, -1]).reshape(B, N, -1)
        preds = torch.stack([head(Z).squeeze(-1) for head in self.heads], 1)
        return preds, {h: {"causal": Z.new_zeros(())} for h in self.horizons}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
