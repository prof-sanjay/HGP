"""
Part B: train and evaluate HICP with one graph configuration.

    python scripts/train.py --config family  --seed 42
    python scripts/train.py --config g12_all --seed 42
    python scripts/train.py --config none    --seed 42
    python scripts/train.py --config lstm    --seed 42     (baseline: per-sensor LSTM, no graph)
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgp.config import DataConfig, ModelConfig, TrainConfig
from hgp.data import load
from hgp.graphs import Assignment
from hgp.train import Trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="family", choices=["family", "g12_all", "g3_all", "shuffled", "road", "road_power", "none", "lstm"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default=DataConfig.path)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(DataConfig.horizons), help="steps ahead; 1 adds the 5-minute horizon")
    ap.add_argument("--graphs", default="graphs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--hidden", type=int, default=ModelConfig.hidden)
    ap.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    ap.add_argument("--layers", type=int, default=ModelConfig.layers, help="LSTM baseline only")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--two-hop", action="store_true")
    ap.add_argument("--physical", action="store_true", help="add a road-adjacency branch to every horizon")
    ap.add_argument("--semantic", action="store_true", help="add a learned similarity-graph branch to every horizon")
    ap.add_argument("--semantic-dim", type=int, default=ModelConfig.semantic_dim, help="node-embedding size of the semantic graph")
    ap.add_argument("--semantic-top-k", type=int, default=ModelConfig.semantic_top_k, help="sources per target in the semantic graph")
    ap.add_argument("--max-windows", type=int, default=None, help="limit windows per split (smoke tests)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dc = DataConfig.from_file(args.data)
    dc.horizons = tuple(args.horizons)
    tc = TrainConfig(graph_config=args.config, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, results_dir=args.results,
                     model=ModelConfig(hidden=args.hidden, dropout=args.dropout, use_gate=not args.no_gate, use_graph=args.config not in ("none", "lstm"),
                                       two_hop=args.two_hop, physical=args.physical, semantic=args.semantic,
                                       semantic_dim=args.semantic_dim, semantic_top_k=args.semantic_top_k,
                                       arch="lstm" if args.config == "lstm" else "hicp", layers=args.layers))
    data = load(dc)
    assignment = Assignment.build("none" if args.config == "lstm" else args.config, dc.horizons, args.graphs, data.road, args.seed)
    trainer = Trainer(data, dc, tc, assignment, torch.device(args.device) if args.device else None, args.max_windows)
    trainer.fit()


if __name__ == "__main__":
    main()
