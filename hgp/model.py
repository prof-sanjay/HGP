"""
HICP: horizon-indexed causal propagation.

    Z     = Encoder(X)                          per-node GRU, shared weights, no cross-sensor mixing
    S_h   = GELU(G_h Z W_h + b_h)               one directed hop over the fixed horizon-h graph
    a_h   = sigmoid(MLP([Z, S_h, e_h]))         gate conditioned on a learned horizon embedding
    H_h   = a_h * S_h + (1 - a_h) * Z
    y_h   = H_h w_h + b_h                       one head per horizon, all horizons in one forward pass
"""

import torch
from torch import nn

from .config import ModelConfig


def propagate(graph: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """graph: sparse [N, N] with graph[i, j] the weight of source j for target i; Z: [B, N, d]."""
    B, N, d = Z.shape
    out = torch.sparse.mm(graph, Z.permute(1, 0, 2).reshape(N, B * d))
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


class Gate(nn.Module):
    def __init__(self, hidden: int, num_horizons: int, embed: int):
        super().__init__()
        self.embed = nn.Embedding(num_horizons, embed)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden + embed, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, Z: torch.Tensor, S: torch.Tensor, k: int) -> torch.Tensor:
        e = self.embed.weight[k].expand(*Z.shape[:2], -1)
        return torch.sigmoid(self.mlp(torch.cat([Z, S, e], -1)))


class HICP(nn.Module):
    def __init__(self, num_features: int, horizons: tuple[int, ...], config: ModelConfig):
        super().__init__()
        self.horizons = tuple(horizons)
        self.config = config
        self.encoder = Encoder(num_features, config.hidden, config.dropout)
        self.propagation = nn.ModuleList([Propagation(config.hidden, config.two_hop, config.dropout) for _ in horizons])
        self.gate = Gate(config.hidden, len(horizons), config.horizon_embed)
        self.heads = nn.ModuleList([nn.Linear(config.hidden, 1) for _ in horizons])

    def forward(self, flow: torch.Tensor, time_feat: torch.Tensor, graphs: dict[int, torch.Tensor | None]):
        """Returns predictions [B, H, N] (standardised) and the mean gate value per horizon."""
        Z = self.encoder(flow, time_feat)
        preds, gates = [], {}
        for k, h in enumerate(self.horizons):
            graph = graphs.get(h) if self.config.use_graph else None
            S = self.propagation[k](Z, graph)
            if graph is None:
                H, gate = Z, Z.new_zeros(())
            elif self.config.use_gate:
                a = self.gate(Z, S, k)
                H, gate = a * S + (1 - a) * Z, a.detach().mean()
            else:
                H, gate = Z + S, Z.new_ones(())
            preds.append(self.heads[k](H).squeeze(-1))
            gates[h] = gate
        return torch.stack(preds, 1), gates


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
